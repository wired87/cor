import os


import dotenv

from app_utils import ARSENAL_PATH
from module_manager.modulator import Modulator
dotenv.load_dotenv()

class ModuleCreator(
    #StateHandler,
    #BaseActor,
):

    """
    Worker for loading, processing and building of single Module
    """

    def __init__(
            self,
            g,
            qfu,
    ):
        """
        Initialize the ModuleCreator instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `G`, `qfu`.
        2. Delegates side effects or helper work through `super().__init__()`, `GUtils()`, `super()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `G`: Graph instance that the workflow reads from or mutates.
        - `qfu`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        super().__init__()
        self.g = g
        self.mmap = []
        self.qfu=qfu
        self.arsenal_struct: list[dict] = None


    def load_sm(self):
        """
        Load sm for the ModuleCreator workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Builds intermediate state such as `new_modules`, `mod_id` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `print()`, `os.listdir()`, `os.path.isdir()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        print("load_sm...")
        new_modules = []
        for i, module_file in enumerate(list(os.listdir(ARSENAL_PATH))):
            if os.path.isdir(os.path.join(ARSENAL_PATH, module_file)) or module_file.startswith("__"):
                continue

            print("load_sm:", module_file)

            if not self.g.G.has_node(module_file):
                mod_id = module_file.split(".")[0].upper()
                new_modules.append(mod_id)

                self.create_modulator(
                    mod_id,
                    i,
                    code=open(
                        os.path.join(
                            ARSENAL_PATH,
                            module_file
                        ),
                        "r",
                        encoding="utf-8"
                    ).read()
                )
        print("sm load successfully modules:", new_modules)


    def main(self, temp_path):
        """
        Run the main ModuleCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `temp_path`.
        2. Branches on validation or runtime state to choose the next workflow path.
        3. Delegates side effects or helper work through `print()`, `os.walk()`, `self.g.G.has_node()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `temp_path`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        print("=========== MODULATOR CREATOR ===========")
        """
        LOOP (TMP) DIR -> CREATE MODULES FORM MEDIA
        """
        # todo load modules form files
        for root, dirs, files in os.walk(temp_path):
            for module in dirs:
                if not self.g.G.has_node(module):
                    self.create_modulator(
                        module,
                    )

            for f in files:
                if not self.g.G.has_node(f):
                    self.create_modulator(
                        f,
                    )
        print("modules updated")


    def create_modulator(self, mid, i, code=None):
        """
        Create modulator for the ModuleCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `mid`, `code`.
        2. Builds intermediate state such as `mref` before applying the main logic.
        3. Delegates side effects or helper work through `print()`, `Modulator()`, `self.g.add_node()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `mid`: Caller-supplied value used during processing.
        - `code`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        try:
            mref = Modulator(
                g=self.g,
                mid=mid,
                qfu=self.qfu,
            )

            # save ref
            self.g.add_node(
                dict(
                    id=mid,
                    type="MODULE",
                    code=code,
                    module_index=i
                )
            )

            print("MODULATORS CREATED")
            mref.module_conversion_process()
        except Exception as e:
            print(f"Err create_modulator: {e}")
        print("create_modulator finished")