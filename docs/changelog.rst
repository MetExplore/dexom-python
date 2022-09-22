.. changelog

CHANGELOG
==========

| This is a changelog for all minor versions of dexom-python before version 1.0.
| Contrary to standard python versioning rules, the different 0.x versions are not fully backwards compatible.
| Standard python versioning will be applied starting with the first stable release, version 1.0.
| This changelog is not exhaustive - it is used to keep track of the main changes.
| In addition to the changes listed here, all versions incorporate bugfixes, new testcases, updated documentation, and other minor modifications.

Versions 0.x
-------------

0.5:
~~~~
| expand and refactor functions for cluster usage
| add DEFAULT_VALUES dictionary for default function parameters
| add main functions to commandline scripts for testing purposes

0.4:
~~~~
| rework cluster functions
| remove prepare_expr_split_gen_list function from gpr_rules module
| generalize apply_gpr function to work for all (currently known) types of GPR-rules

0.3:
~~~~
| resolve import problems and unify import behavior by renaming modules
| expand documentation to cover more functions
| create create_enum_variables function
| modify output behavior - solutions are no longer saved within the enumeration functions
| correct gpr_rules behavior

0.2:
~~~~
| create gpr_rules.py
| add new variable implementation for imat
| create read_model, check_model_options, Fischer_groups functions
| remove get_binary_sol, get_obj_value_from_binary functions
| add documentation
| add unit tests
