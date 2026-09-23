# EEC27 Koopman Tutorial

This repository holds the code for the tutorial paper: *"A TCLab tutorial on Koopman control methods"*.

> [!warning]
> Update the title once the paper is submitted!

## Prerequisites

- `MATLAB`: [MATLAB R2026a](https://uk.mathworks.com/products/matlab.html)
- `Python`: [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

1. Clone this repository or download the code as a ZIP archive and extract it to a folder of your choice.

2. `MATLAB`

   1. To initalize the workspace run the following script

        ```matlab
        run('matlab/initWorkspace.m')
        ```

   2. Run experiemtns with

        ```matlab
        run('matlab/runSimulation.m')
        ```

3. `Python`

   1. From the project root, set up the environment by running

       ```bash
       uv sync
       ```

       This creates a `.venv` folder and installs the pinned Python version, together with all dependencies.

   2. Run commands inside the environment with `uv run`,

       ```bash
       uv run python ./LTI_embedding.py
       ```

       or alternatively, activate the environment as usual

       ```bash
       source .venv/bin/activate
       ```

## License

This project is licensed under the MIT license.

## Support

For any questions or issues related to this code, please contact the author:

- Alex Erdin: [alex.erdin@tuhh.de](mailto:alex.erdin@tuhh.de)
- Lea Bold: [lea.bold@tu-ilmenau.de](mailto:lea.bold@tu-ilmenau.de)

We appreciate any feedback, bug reports, or suggestions for improvements.
