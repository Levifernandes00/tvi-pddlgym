# tvi-pddlgym
Using Gabriel Crispino Code, implemented topological value iteration.

# vi-pddlgym
This repository contains a sample implementation that can solve MDPs represented by PDDLGym environments using the Value Iteration algorithm.

This implementation uses the function `get_successor_states` imported from PDDLGym's `core` module (`pddlgym.core`).
Since this feature is currently available in PDDLGym's repository but not in its latest pypi release as of now,
to use it you'll need to either clone the repository and install it locally or install it via pip by pointing to the repository.
You can do the former by setting up a virtual env ([see here](https://github.com/tomsilver/pddlgym#installing-from-source-if-you-want-to-make-changes-to-pddlgym)) or the latter by running the following:

`$ pip install git+https://github.com/tomsilver/pddlgym`

## Usage
For usage instructions, run `python main.py --help` in the repository's root folder

The following command example can be used to solve the `PDDLEnvBlocks-v0` environment, simulate an episode with the resulting policy and render each encountered observation:

`$ python main.py --env PDDLEnvBlocks-v0 --simulate --render_and_save`
