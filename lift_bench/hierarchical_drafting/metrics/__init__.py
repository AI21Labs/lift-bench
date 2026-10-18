import importlib
import pkgutil

import lift_bench.hierarchical_drafting.metrics as package

package_dir = package.__path__[0]
for _, module_name, _ in pkgutil.walk_packages([package_dir], prefix=package.__name__ + "."):
    importlib.import_module(module_name)
