
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

import os

import law
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox

from mtt.tasks.inference.inference_base import InferenceBaseTask

logger = law.logger.get_logger(__name__)


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
        regions = self.input()["datacards"].keys()
        cards = []
        for reg in regions:
            cards.append(self.input()["datacards"][reg]['card'].path)
        if "default" in self.inference_model or "zp" in self.inference_model:
            logger.info("Using merged datacards as inputs.")
        else:
            raise ValueError("Only 'default' inf model currently implemented.")
        input_datacards = cards
        output_card = self.output()["combined_card"].path
        if not os.path.exists(os.path.dirname(output_card)):
            logger.warning("Output path didn't exist yet.")
            os.makedirs(os.path.dirname(output_card))

        logger.warning("Make sure the cards are assigned to the correct cats!")

        command_to_run = "combineCards.py"
        for i, card in enumerate(input_datacards):
            command_to_run += f" {self.inference_model_inst.categories[i].name}={card}"

        command_to_run += f" &> {output_card}"
        command_to_run += " -h" if self.print_help else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["log"].dump("\n".join(outp), formatter="text")
