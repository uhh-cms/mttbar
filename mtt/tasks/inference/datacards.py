
"""
A task that calls the combineCards combine command.

combine_datacards() {
    local args24=(
        mu_sr1_24=${INPUT_DIR}/datacard__mu_sr1_24.txt
        mu_sr2_24=${INPUT_DIR}/datacard__mu_sr2_24.txt
        mu_cr1_24=${INPUT_DIR}/datacard__mu_cr1_24.txt
        mu_cr2_24=${INPUT_DIR}/datacard__mu_cr2_24.txt
        el_sr1_24=${INPUT_DIR}/datacard__el_sr1_24.txt
        el_sr2_24=${INPUT_DIR}/datacard__el_sr2_24.txt
        el_cr1_24=${INPUT_DIR}/datacard__el_cr1_24.txt
        el_cr2_24=${INPUT_DIR}/datacard__el_cr2_24.txt
    )
    local args25=(
        mu_sr1_25=${INPUT_DIR}/datacard__mu_sr1_25.txt
        mu_sr2_25=${INPUT_DIR}/datacard__mu_sr2_25.txt
        mu_cr1_25=${INPUT_DIR}/datacard__mu_cr1_25.txt
        mu_cr2_25=${INPUT_DIR}/datacard__mu_cr2_25.txt
        el_sr1_25=${INPUT_DIR}/datacard__el_sr1_25.txt
        el_sr2_25=${INPUT_DIR}/datacard__el_sr2_25.txt
        el_cr1_25=${INPUT_DIR}/datacard__el_cr1_25.txt
        el_cr2_25=${INPUT_DIR}/datacard__el_cr2_25.txt
    )
    local args_merged=(
        mu_sr1=${INPUT_DIR}/datacard__mu_sr1.txt
        mu_sr2=${INPUT_DIR}/datacard__mu_sr2.txt
        mu_cr1=${INPUT_DIR}/datacard__mu_cr1.txt
        mu_cr2=${INPUT_DIR}/datacard__mu_cr2.txt
        el_sr1=${INPUT_DIR}/datacard__el_sr1.txt
        el_sr2=${INPUT_DIR}/datacard__el_sr2.txt
        el_cr1=${INPUT_DIR}/datacard__el_cr1.txt
        el_cr2=${INPUT_DIR}/datacard__el_cr2.txt
    )
    if [[ "$YEAR" == "2425" ]]; then
        args=("${args24[@]}" "${args25[@]}")
    elif [[ "$YEAR" == "24" ]]; then
        args=("${args24[@]}")
    elif [[ "$YEAR" == "25" ]]; then
        args=("${args25[@]}")
    elif [[ "$YEAR" == "merged" ]]; then
        args=("${args_merged[@]}")
    else
        echo "Invalid YEAR specified: $YEAR. Must be '24', '25', '2425', or 'merged'."
        exit 1
    fi
    local cmd="combineCards.py ${args[*]} &> ${DATACARD}"
    run_cmd "$cmd" "${LOG_DIR}/combine_datacards_${FIT_VERSION}__${YEAR}.log"
}
"""
from __future__ import annotations

import os

import law
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox

from mtt.tasks.inference.inference_base import InferenceBaseTask

logger = law.logger.get_logger(__name__)


def build_combine_cards_cmd(
    cards: dict[str, str],
    output_card: str,
    log_path: str,
    print_help: bool = False,
) -> str:
    """
    Builds the shell command that merges per-category datacards with ``combineCards.py``.

    *cards* maps category names to datacard paths. combineCards prints the merged card to stdout, so
    stdout is redirected into *output_card* and stderr (warnings, errors) into *log_path*, to keep the
    two apart. This matters as anything on stderr would otherwise end up inside the datacard.
    """
    cmd = "combineCards.py"
    cmd += "".join(f" {name}={path}" for name, path in cards.items())
    cmd += " -h" if print_help else ""
    cmd += f" > {output_card} 2> {log_path}"
    return cmd


class CombineDatacards(
    InferenceBaseTask,
):
    sandbox = dev_sandbox(law.config.get("analysis", "combine_sandbox"))
    resolution_task_cls = CreateDatacards

    @property
    def card_name(self):
        return "combined_card"

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        parts.insert_after("inf_model", "combined_datacard", self.card_name)

        return parts

    def output(self):
        return {
            "combined_card": self.target("datacard.txt"),
            "log": self.target("combine_cards.log"),
        }

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        if not ("default" in self.inference_model or "zp" in self.inference_model):
            raise ValueError(
                f"inference model '{self.inference_model}' not supported, its name must contain "
                "'default' or 'zp' (only these use merged datacards so far)",
            )
        logger.info("Using merged datacards as inputs.")

        # the keys of the CreateDatacards output are the category names, so use them directly
        # instead of relying on the order of the categories in the inference model
        datacards = self.input()["datacards"]
        cards = {cat_name: datacards[cat_name]["card"].path for cat_name in datacards.keys()}

        output_card = self.output()["combined_card"].path
        log_path = self.output()["log"].path
        os.makedirs(os.path.dirname(output_card), exist_ok=True)

        command_to_run = build_combine_cards_cmd(cards, output_card, log_path, print_help=self.print_help)

        # stdout and stderr of combineCards.py are redirected to files, so run_command only captures its
        # own header; stderr is already in the log file and is appended to the header below
        p, outp = self.run_command(command_to_run, echo=True)
        with open(log_path) as f:
            stderr = f.read()
        self.output()["log"].dump(f"{outp}\n{stderr}", formatter="text")

        if os.path.getsize(output_card) == 0:
            # note: safe_output removes all outputs (including the log) when raising, so include stderr here
            raise RuntimeError(f"combineCards.py created an empty datacard {output_card}, stderr:\n{stderr}")
