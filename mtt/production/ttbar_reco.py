# coding: utf-8

"""
Column production methods related to ttbar mass reconstruction.
"""
import itertools
import math

from law.util import human_duration

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT, attach_coffea_behavior

from mtt.config.categories import add_categories_production
from mtt.util import iter_chunks
from mtt.production.util import (
    ak_argcartesian, ak_arg_grouped_combinations, lv_xyzt, lv_mass, lv_sum,
)
from mtt.production.lepton import choose_lepton
from mtt.production.neutrino import neutrino_candidates
from mtt.production.ttbar_gen import ttbar_gen
from mtt.profiling_tools import Profiler

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        choose_lepton, neutrino_candidates, category_ids,
        "channel_id",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "FatJetTopTagDeltaRLepton.pt", "FatJetTopTagDeltaRLepton.eta",
        "FatJetTopTagDeltaRLepton.phi", "FatJetTopTagDeltaRLepton.mass",
        "FatJetTopTagDeltaRLepton.msoftdrop",
    },
    produces={
        choose_lepton, neutrino_candidates, category_ids,
        "TTbar.*",
    },
)
def ttbar(
    self: Producer,
    events: ak.Array,
    # algorithm tweaks
    merge_mode="eager",
    # profiling/reporting options
    profile_memory=False,
    profile_time=True,
    verbose_level=1,
    **kwargs,
) -> ak.Array:
    """
    Reconstruct the ttbar pair in the semileptonic decay mode.
    This is done by evaluating all possibilities of assigning the lepton,
    jets, and neutrino to the hadronic and leptonic legs of the decay,
    in terms of a chi2 metric. The configuration with the lowest
    chi2 out of all possibilities is selected.

    Parameters:
      - *n_jet_max*: limit the number of jets per event to at most this number
      - *n_jet_lep_range*: minimum and maximum number of jets that can be assigned to the leptonic top decay
      - *n_jet_had_range*: minimum and maximum number of jets that can be assigned to the hadronic top decay
      - *n_jet_ttbar_range*: minimum and maximum number of jets that can be assigned to the ttbar decay overall
        (if *None* or omitted, will be inferred from *n_jet_lep_range* and *n_jet_had_range*)

    The parameter values can be provided as keyword arguments or via a configuration entry:

    .. code-block:: python

        cfg.x.ttbar_reco_settings = {
            "n_jet_max": 9,
            "n_jet_lep_range": (1, 2),
            "n_jet_had_range": (2, 6),
            "n_jet_ttbar_range": (3, 6),
        }

    """
    # -- obtain settings from kwargs or config
    settings = self.config_inst.x.ttbar_reco_settings
    for setting in ("n_jet_max", "n_jet_lep_range", "n_jet_had_range", "n_jet_ttbar_range", "max_chunk_size"):
        setting_value = kwargs.get(setting, settings.get(setting, None))
        if setting_value is None and setting != "n_jet_ttbar_range":
            raise ValueError(f"setting '{setting}' must be provided via config or kwargs.")
        settings[setting] = setting_value

    # store settings in local variables
    n_jet_max = settings["n_jet_max"]
    n_jet_ttbar_range = settings["n_jet_ttbar_range"]
    n_jet_lep_range = settings["n_jet_lep_range"]
    n_jet_had_range = settings["n_jet_had_range"]
    max_chunk_size = settings["max_chunk_size"]

    # resolve max chunk size if dynamic
    if callable(max_chunk_size):
        max_chunk_size = max_chunk_size(self.dataset_inst)

    # infer missing settings
    if n_jet_ttbar_range is None:
        n_jet_ttbar_range = (
            n_jet_lep_range[0] + n_jet_had_range[0],
            n_jet_lep_range[1] + n_jet_had_range[1],
        )

    # check range values are meaningful
    for range_par in n_jet_lep_range, n_jet_had_range, n_jet_ttbar_range:
        assert len(range_par) == 2
        assert range_par[0] >= 1
        assert range_par[1] <= n_jet_max

    # validate merging mode
    assert merge_mode in ("eager", "lazy"), f"invalid merge_mode '{merge_mode}'"
    self.task.publish_message(f"merge mode is '{merge_mode}'")

    # load coffea behaviors for simplified arithmetic with vectors
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")
    events["FatJetTopTagDeltaRLepton"] = ak.with_name(events.FatJetTopTagDeltaRLepton, "PtEtaPhiMLorentzVector")
    events = attach_coffea_behavior(
        events,
        collections={"GenPart": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}},
    )

    # reconstruct neutrino candidates
    events = self[neutrino_candidates](events, **kwargs)
    nu_cands = events["NeutrinoCandidates"]

    # get leptons
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # -- AK8 jets: only top-tagged jets well separated from main lepton
    topjet = events.FatJetTopTagDeltaRLepton

    # tag events as boosted if there is at least one AK8 jet
    is_boosted = ak.fill_none(ak.num(topjet, axis=1) >= 1, False)

    # -- AK4 jets: only keep the first `n_jet_max` jets per event
    jet = events.Jet[ak.local_index(events.Jet) < n_jet_max]

    # well separated from AK8 jets (deltaR >= 1.2)
    delta_r_jet_topjet = ak.min(jet.metric_table(topjet), axis=2)
    jet_isolated = ak.fill_none(delta_r_jet_topjet > 1.2, True)
    jet = jet[jet_isolated]

    # split jet array into boosted/resolved cases
    jet_resolved = ak.where(
        is_boosted,
        [[]],
        jet,
    )
    jet_boosted = ak.where(
        ~is_boosted,
        [[]],
        jet,
    )

    # pack arrays to ensure contiguous memory
    # and trim unreachable entries
    jet_resolved = ak.to_packed(jet_resolved)
    jet_boosted = ak.to_packed(jet_boosted)

    # make lorentz vectors
    topjet_lv = lv_xyzt(topjet)
    jet_lv = {
        "resolved": lv_xyzt(jet_resolved),
        "boosted": lv_xyzt(jet_boosted),
    }
    lepton_lv = lv_xyzt(ak.unflatten(lepton, counts=1))
    nu_cands_lv = lv_xyzt(nu_cands)

    # -- handle combinatorics

    # helper functions for choosing best-chi2 combination
    # after *each round* of combinatorics
    def merge_sequential(best_results, new_results):
        if best_results is None:
            best_results = new_results
        else:
            # update results if chi2 decreases
            new_chi2, best_chi2 = new_results["chi2"], best_results["chi2"]
            update = (new_chi2 < best_chi2)
            # handle missing values
            update = ak.fill_none(
                ak.where(ak.is_none(update), ~ak.is_none(new_chi2), update),
                False,
            )
            best_results = {
                res_var: ak.to_packed(ak.where(
                    update,
                    new_results[res_var],
                    best_results[res_var],
                ))
                for res_var in best_results
            }

        return best_results

    # helper functions for choosing best-chi2 combination
    # after *all rounds* of combinatorics
    def merge_all(comb_results):
        comb_specs = list(comb_results)
        result_vars = list(comb_results[comb_specs[0]])
        comb_results_merged = {
            result_var: ak.to_packed(ak.concatenate(
                [
                    ak.singletons(
                        comb_result[result_var],
                        axis=0,
                    )
                    for comb_result in comb_results.values()
                ],
                axis=1,
            ))
            for result_var in result_vars
        }

        # choose combination with smallest chi2
        best_comb_idx = ak.argmin(
            comb_results_merged["chi2"],
            axis=1,
            keepdims=True,
        )
        best_results = {
            result_var: ak.firsts(comb_results_merged[result_var][best_comb_idx])
            for result_var in comb_results_merged
        }

        return best_results

    # timer contexts for monitoring runtime
    def profile_task(name, min_verbose_level=0, indent_level=0, **kwargs):
        """
        Create a context manager for profiling and reporting
        execution status of code sections.
        """
        return Profiler(
            task_name=name,
            # report on task completion
            msg_func=(
                self.task.publish_message
                if verbose_level >= min_verbose_level
                else None
            ),
            # indent messages for structured output
            indent_str="  " * indent_level,
            n_cols_text=max(50, 80 - 2 * indent_level),
            # trigger manual garbage collection after task
            gc_on_exit=True,
            # enable/disable profiling metrics
            prof_mem=profile_memory,
            prof_time=profile_time,
            # allow users to supply other kwargs
            **kwargs,
        )

    # helper function for reconstructing ttbar decay
    # builds all possible ways of arranging combinations into
    # groups with fixed sizes (`n_jets`)
    def ttbar_combinatorics(jet_lv, topjet_lv, topjet_msoftdrop, lepton_lv, nu_cands_lv, n_jets, regime="resolved"):
        """
        Reconstruct the leptonically and hadronically decaying top quarks
        from combinations of final-state objects.

        These should be provided as arrays of Lorentz vectors as follows:
          * AK4 jets (`jet_lv`),
          * top-tagged AK8 jets (`topjet_lv`) (boosted regime only),
          * softdrop-mass of top-tagged AK8 jets (`topjet_msoftdrop`) (boosted regime only),
          * leptons (`lepton_lv`),
          * neutrino candidates (`nu_cands_lv`).

        The `regime` argument can be either "resolved" or "boosted".

        In the `resolved` regime, the decays of both top quarks are reconstructed
        from AK4 jets. In this case, `n_jets` should be a tuple (`n_jet_lep`, `n_jet_had`),
        indicating how many jets should be assigned to the leptonically  and hadronically
        decaying top quark, in that order.

        In the `boosted` regime, the hadronically decaying top quark is identified with
        the (highest-pt) top-tagged AK8 jet, and only the decay of the leptonically decaying
        top quark is reconstructed from AK4 jets. In this case, `n_jets` should be a
        tuple with a single element (`n_jet_lep`), indicating how many jets should be mapped
        to the leptonically decaying top quark.
        """
        # validate inputs
        if regime == "resolved":
            assert len(n_jets) == 2, (
                f"`n_jets` must have length 2 (lep, had) "
                f"for resolved regime, got {len(n_jets)}"
            )
            is_boosted_regime = False
            n_jet_lep, n_jet_had = n_jets

        elif regime == "boosted":
            assert len(n_jets) == 1, (
                f"`n_jets` must have length 1 (lep) "
                f"for resolved regime, got {len(n_jets)}"
            )
            is_boosted_regime = True
            n_jet_lep = n_jets[0]

        else:
            assert False, f"unknown regime: {regime}"

        # jet index combinations for all possible groupings of jets
        # into two distinct sets with sizes `n_jet_lep` and `n_jet_had`
        with profile_task(
            f"build jet index combinations {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            jet_idx_combs = ak_arg_grouped_combinations(
                jet_lv,
                group_sizes=n_jets,
                axis=1,
            )
            if is_boosted_regime:
                jet_idx_comb_lep = jet_idx_combs[0]
            else:
                jet_idx_comb_lep, jet_idx_comb_had = jet_idx_combs

        # index configurations for all choices of lepton, neutrino candidate,
        # and jet combination mapped to leptonic decay
        with profile_task(
            f"build lepton/neutrino/jet combination product {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            lnu_jetcomb_idx_prod = ak_argcartesian(
                lepton_lv,
                nu_cands_lv,
                jet_idx_comb_lep[0],
            )

        # retrieve chi2 parameters from config
        chi2_pars = self.config_inst.x.chi2_parameters[regime]

        # -- set up hypotheses for leptonically decaying tops

        with profile_task(
            f"sum over leptonic jets {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            jet_sum_lep = lv_sum((
                jet_lv[jet_idx]
                for jet_idx in jet_idx_comb_lep
            ))

        with profile_task(
            f"construct leptonic top hypotheses {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            hyp_top_lep = (
                lepton_lv[lnu_jetcomb_idx_prod[0]] +
                nu_cands_lv[lnu_jetcomb_idx_prod[1]] +
                jet_sum_lep[lnu_jetcomb_idx_prod[2]]
            )

        # calculate hypothesis chi2 scores
        with profile_task(
            f"calculate chi2 for hypotheses {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            hyp_top_lep_chi2 = ((hyp_top_lep.mass - chi2_pars.m_lep) / chi2_pars.s_lep) ** 2
            # store number of jets matched to leptonic decay
            hyp_n_jet_lep = ak.ones_like(hyp_top_lep_chi2, dtype=np.uint8) * n_jet_lep

        # -- set up hypotheses for hadronically decaying tops

        if regime == "resolved":
            with profile_task(
                f"sum over hadronic jets {n_jets}",
                indent_level=2,
                min_verbose_level=2,
            ):
                jet_sum_had = lv_sum((
                    jet_lv[jet_idx]
                    for jet_idx in jet_idx_comb_had
                ))

            with profile_task(
                f"construct hadronic top hypotheses {n_jets}",
                indent_level=2,
                min_verbose_level=2,
            ):
                hyp_top_had = jet_sum_had[lnu_jetcomb_idx_prod[2]]

            with profile_task(
                f"calculate chi2 for hypotheses {n_jets}",
                indent_level=2,
                min_verbose_level=2,
            ):
                hyp_top_had_chi2 = ((hyp_top_had.mass - chi2_pars.m_had) / chi2_pars.s_had) ** 2
                # store number of jets matched to hadronic decay
                hyp_n_jet_had = ak.ones_like(hyp_top_had_chi2, dtype=np.uint8) * n_jet_had

        # reduce hypotheses based on minimal total chi2 score
        with profile_task(
            f"select best hypothesis {n_jets}",
            indent_level=2,
            min_verbose_level=2,
        ):
            if is_boosted_regime:
                hyp_top_chi2 = hyp_top_lep_chi2
            else:
                hyp_top_chi2 = hyp_top_lep_chi2 + hyp_top_had_chi2

            hyp_top_chi2_argmin = ak.argmin(hyp_top_chi2, axis=1, keepdims=True)

            # store final leptonic top
            top_lep = ak.firsts(hyp_top_lep[hyp_top_chi2_argmin])
            n_jet_lep = ak.firsts(hyp_n_jet_lep[hyp_top_chi2_argmin])
            top_lep_chi2 = ak.firsts(hyp_top_lep_chi2[hyp_top_chi2_argmin])

            # get mapped jet indices
            jetcomb_idx = lnu_jetcomb_idx_prod[2][hyp_top_chi2_argmin]
            jet_idxs_lep = ak.concatenate([
                jet_idx[jetcomb_idx]
                for jet_idx in jet_idx_comb_lep
            ], axis=1)

            # store final hadronic top
            if is_boosted_regime:
                top_had = ak.firsts(topjet_lv)
                top_had_chi2 = ((ak.firsts(topjet_msoftdrop) - chi2_pars.m_had) / chi2_pars.s_had) ** 2
                n_jet_had = ak.zeros_like(n_jet_lep, dtype=np.uint8)
                # array of empty lists
                jet_idxs_had = ak.values_astype(
                    ak.drop_none(
                        ak.mask(
                            hyp_top_chi2_argmin,
                            ak.zeros_like(hyp_top_chi2_argmin, dtype=bool),
                            valid_when=True,
                        ),
                    ),
                    np.uint8,
                )
            else:
                top_had = ak.firsts(hyp_top_had[hyp_top_chi2_argmin])
                top_had_chi2 = ak.firsts(hyp_top_had_chi2[hyp_top_chi2_argmin])
                n_jet_had = ak.firsts(hyp_n_jet_had[hyp_top_chi2_argmin])
                jet_idxs_had = ak.concatenate([
                    jet_idx[jetcomb_idx]
                    for jet_idx in jet_idx_comb_had
                ], axis=1)

            # store final chi2 score
            chi2 = top_had_chi2 + top_lep_chi2

        return {
            "top_had": top_had,
            "top_lep": top_lep,
            "n_jet_had": n_jet_had,
            "n_jet_lep": n_jet_lep,
            "jet_idxs_had": jet_idxs_had,
            "jet_idxs_lep": jet_idxs_lep,
            "top_had_chi2": top_had_chi2,
            "top_lep_chi2": top_lep_chi2,
            "chi2": chi2,
        }

    # threads over all allowed multiplicities of jets
    # assigned to the leptonic and hadronic top quark decays
    def main_loop(
        jet_lv,
        topjet_lv,
        topjet_msoftdrop,
        lepton_lv,
        nu_cands_lv,
        regime,
    ):
        # obtain all possible jet multiplicities for leptonic top decay
        iter_n_jets = [range(n_jet_lep_range[0], n_jet_lep_range[1] + 1)]

        # resolved case: extend iterator list to cover hadronic top decay
        if regime == "resolved":
            iter_n_jets.append(range(n_jet_had_range[0], n_jet_had_range[1] + 1))

        # loop over all jet multiplicities and collect results
        comb_results = None if merge_mode == "eager" else {}
        total_merge_time = 0.
        for n_jets in itertools.product(*iter_n_jets):

            # skip if too few/many jets mapped to overall ttbar decay
            n_jet_ttbar = sum(n_jets)
            if (
                regime == "resolved" and
                not (n_jet_ttbar_range[0] <= n_jet_ttbar <= n_jet_ttbar_range[1])
            ):
                continue

            # do the reconstruction
            with profile_task(
                f"{regime} reco with {n_jets} jets",
                indent_level=1,
                min_verbose_level=1,
            ):
                results = ttbar_combinatorics(
                    jet_lv,
                    topjet_lv,
                    topjet_msoftdrop,
                    lepton_lv,
                    nu_cands_lv,
                    n_jets=n_jets,
                    regime=regime,
                )

            if merge_mode == "eager":
                # merge results immediately
                with profile_task(
                    "merge to current best results",
                    indent_level=1,
                    min_verbose_level=2,
                ) as t:
                    comb_results = merge_sequential(comb_results, results)

                if profile_time:
                    total_merge_time += t.duration
            else:
                # add results to a dict to be merged later
                comb_results[n_jets] = results

        # finally merge comb_results across all values of `n_jets`
        if merge_mode == "lazy":
            with profile_task(
                "merge results for all combination multiplicities",
                indent_level=1,
                min_verbose_level=2,
            ):
                comb_results = merge_all(comb_results)

        # if merging was done eagerly, report on cumulated time spent
        if profile_time and merge_mode == "eager" and verbose_level >= 2:
            self.task.publish_message(
                "total time spent merging: "
                f"{human_duration(seconds=total_merge_time)}",
            )

        return comb_results

    def apply_chunked(func, arrays, max_chunk_size, **kwargs):
        """
        Apply function `func` to identically-sized `arrays` in a chunked way
        and merge the results. The `func` should return an `ak.Array` or a
        dictionary containing `ak.Array` objects.
        """
        result = None
        size = len(arrays[0])
        n_chunks = max(1, int(math.ceil(size / max_chunk_size)))
        self.task.publish_message(
            f"processing {size} events in {n_chunks} sub-chunks",
        )
        for i_chk, arrays_chk in enumerate(
            iter_chunks(*arrays, max_chunk_size=max_chunk_size),
        ):
            with profile_task(
                f"processing sub-chunk {i_chk + 1}/{n_chunks}",
            ):

                new_result = func(*arrays_chk, **kwargs)

                if result is None:
                    result = new_result
                elif isinstance(result, ak.Array):
                    result = ak.to_packed(ak.concatenate([result, new_result], axis=0))
                elif isinstance(result, dict):
                    assert set(result) == set(new_result)
                    result = {
                        key: ak.to_packed(ak.concatenate([result[key], new_result[key]], axis=0))
                        for key in result
                    }

        return result

    # apply main loop in both regimes
    comb_results = {}

    for regime in ("resolved", "boosted"):
        comb_results[regime] = apply_chunked(
            main_loop,
            (
                jet_lv[regime],
                topjet_lv,
                topjet.msoftdrop,
                lepton_lv,
                nu_cands_lv,
            ),
            regime=regime,
            max_chunk_size=max_chunk_size,
        )

    # merge regimes
    with profile_task("merge resolved and boosted reconstructions"):
        result_keys = set(comb_results["resolved"])
        assert result_keys == set(comb_results["boosted"]), \
            "resolved and boosted result keys mismatched"
        comb_results = {
            key: ak.to_packed(ak.where(
                is_boosted,
                comb_results["boosted"][key],
                comb_results["resolved"][key],
            ))
            for key in result_keys
        }

    # store final top
    top_had = lv_mass(comb_results["top_had"])
    top_lep = lv_mass(comb_results["top_lep"])

    # store mapped jet indices and counts
    jet_idxs_had = comb_results["jet_idxs_had"]
    jet_idxs_lep = comb_results["jet_idxs_lep"]
    n_jet_had = comb_results["n_jet_had"]
    n_jet_lep = comb_results["n_jet_lep"]

    # store final chi2 scores
    top_had_chi2 = comb_results["top_had_chi2"]
    top_lep_chi2 = comb_results["top_lep_chi2"]
    chi2 = comb_results["chi2"]

    # sum over top quarks to form ttbar system
    ttbar = lv_mass(top_had + top_lep)

    # -- calculate cos(theta*)

    # boost lepton + leptonic top quark to ttbar rest frame
    top_lep_ttrest = top_lep.boost(-ttbar.boostvec)

    # get cosine from three-vector dot product and magnitudes
    cos_theta_star = ttbar.pvec.dot(top_lep_ttrest.pvec) / (ttbar.pvec.p * top_lep_ttrest.pvec.p)
    abs_cos_theta_star = abs(cos_theta_star)

    # -- calculate energy of hadronic and leptonic top
    top_had_energy = np.sqrt((top_had.mass) ** 2 + (top_had.pt) ** 2)
    top_lep_energy = np.sqrt((top_lep.mass) ** 2 + (top_lep.pt) ** 2)

    # write out columns
    for var in ("pt", "eta", "phi", "mass"):
        events = set_ak_column(events, f"TTbar.top_had_{var}", ak.fill_none(getattr(top_had, var), EMPTY_FLOAT))
        events = set_ak_column(events, f"TTbar.top_lep_{var}", ak.fill_none(getattr(top_lep, var), EMPTY_FLOAT))
        events = set_ak_column(events, f"TTbar.{var}", ak.fill_none(getattr(ttbar, var), EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.top_had_energy", ak.fill_none(top_had_energy, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.top_lep_energy", ak.fill_none(top_lep_energy, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.n_jet_had", ak.fill_none(n_jet_had, -1))
    events = set_ak_column(events, "TTbar.n_jet_lep", ak.fill_none(n_jet_lep, -1))
    events = set_ak_column(events, "TTbar.n_jet_sum", ak.fill_none(n_jet_lep + n_jet_had, -1))
    events = set_ak_column(events, "TTbar.jet_idxs_had", ak.drop_none(jet_idxs_had))
    events = set_ak_column(events, "TTbar.jet_idxs_lep", ak.drop_none(jet_idxs_lep))
    events = set_ak_column(events, "TTbar.chi2_had", ak.fill_none(top_had_chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.chi2_lep", ak.fill_none(top_lep_chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.chi2", ak.fill_none(chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.cos_theta_star", ak.fill_none(cos_theta_star, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.abs_cos_theta_star", ak.fill_none(abs_cos_theta_star, EMPTY_FLOAT))

    if self.dataset_inst.is_mc:
        # run producer to obtain gen-level ttbar
        events = self[ttbar_gen](events, **kwargs)
        gen_ttbar = events["GenTTbar"]
        gen_part = events["GenPart"]
        gen_top_1 = ak.firsts(gen_part[gen_ttbar.top_1])
        gen_top_2 = ak.firsts(gen_part[gen_ttbar.top_2])

        # match gen-level tops to hadronic and leptonic reco tops
        dr_top_had_gen_top_1 = top_had.delta_r(gen_top_1)
        dr_top_lep_gen_top_2 = top_lep.delta_r(gen_top_2)
        dr_top_had_gen_top_2 = top_had.delta_r(gen_top_2)
        dr_top_lep_gen_top_1 = top_lep.delta_r(gen_top_1)

        dr_sum_match_12 = dr_top_had_gen_top_1 + dr_top_lep_gen_top_2
        dr_sum_match_21 = dr_top_had_gen_top_2 + dr_top_lep_gen_top_1

        top_had_first = dr_sum_match_12 < dr_sum_match_21
        gen_top_had = ak.where(
            top_had_first,
            gen_top_1,
            gen_top_2,
        )
        gen_top_lep = ak.where(
            top_had_first,
            gen_top_2,
            gen_top_1,
        )
        gen_w_had = gen_part[ak.where(
            top_had_first,
            gen_ttbar.w_1,
            gen_ttbar.w_2,
        )]
        gen_w_lep = gen_part[ak.where(
            top_had_first,
            gen_ttbar.w_2,
            gen_ttbar.w_1,
        )]
        gen_ttbar_lv = lv_mass(gen_top_had) + lv_mass(gen_top_lep)

        # -- calculate gen-level cos(theta*)

        # boost lepton + leptonic top quark to ttbar rest frame
        gen_top_lep_ttrest = lv_mass(gen_top_lep).boost(-gen_ttbar_lv.boostvec)
        gen_cos_theta_star = (
            gen_ttbar_lv.pvec.dot(gen_top_lep_ttrest.pvec) / (gen_ttbar_lv.pvec.p * gen_top_lep_ttrest.pvec.p)
        )
        gen_abs_cos_theta_star = abs(gen_cos_theta_star)

        dr_gen_top_had = lv_mass(gen_top_had).delta_r(top_had)
        dr_gen_top_lep = lv_mass(gen_top_lep).delta_r(top_lep)
        dr_gen_ttbar = gen_ttbar_lv.delta_r(ttbar)

        def set_ak_column_ef(events, name, value):
            """Set float column, replacing missing values by EMPTY_FLOAT."""
            return set_ak_column(events, name, ak.fill_none(value, EMPTY_FLOAT))

        for var in ("pt", "eta", "phi", "mass"):
            events = set_ak_column_ef(events, f"TTbar.gen_top_had_{var}", getattr(gen_top_had, var))
            events = set_ak_column_ef(events, f"TTbar.gen_top_lep_{var}", getattr(gen_top_lep, var))
            events = set_ak_column_ef(events, f"TTbar.gen_w_had_{var}", getattr(gen_w_had, var))
            events = set_ak_column_ef(events, f"TTbar.gen_w_lep_{var}", getattr(gen_w_lep, var))
            events = set_ak_column_ef(events, f"TTbar.gen_{var}", getattr(gen_ttbar_lv, var))
        events = set_ak_column_ef(events, "TTbar.gen_top_had_delta_r", dr_gen_top_had)
        events = set_ak_column_ef(events, "TTbar.gen_top_lep_delta_r", dr_gen_top_lep)
        events = set_ak_column_ef(events, "TTbar.gen_delta_r", dr_gen_ttbar)
        events = set_ak_column_ef(events, "TTbar.gen_cos_theta_star", gen_cos_theta_star)
        events = set_ak_column_ef(events, "TTbar.gen_abs_cos_theta_star", gen_abs_cos_theta_star)

    # recalculate category ids with ttbar information
    events = self[category_ids](events, **kwargs)

    return events


@ttbar.init
def ttbar_init(self: Producer) -> None:
    # add production categories to config
    if not self.config_inst.get_aux("has_categories_production", False):
        add_categories_production(self.config_inst)
        self.config_inst.x.has_categories_production = True

    if hasattr(self, "dataset_inst") and self.dataset_inst.is_mc:
        self.uses |= {ttbar_gen}
        self.produces |= {ttbar_gen}
