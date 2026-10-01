# EEC27 Koopman Tutorial

This repository holds the MATLAB code for the tutorial paper: *"A TCLab tutorial on Koopman control methods"*.

> [!warning]
> Update the title once the paper is submitted!

## Prerequisites

- [MATLAB R2026a](https://uk.mathworks.com/products/matlab.html)

## Installation

1. Clone this repository or download the code as a ZIP archive and extract it to a folder of your choice.

2. To initalize the workspace run the following script

    ```matlab
    initWorkspace.m
    ```

3. Run the simulation with

    ```matlab
    learnTCLab.m
    ```

## How does it work?

```mermaid
flowchart TB
    subgraph Learning
        direction LR
        Data@{shape: lean-l} -->|"$$\Bigl(x^+,x,u\Bigr)$$"| Scale1[Scaling]
        Scale1 -->|"$$\Bigl(\hat{x}^+,\hat{x},\hat{u}\Bigr)$$"| Lift1[Lifting] 
        Lift1 -->|"$$\Bigl(\varphi(\hat{x}^+),\varphi(\hat{x}),\psi(\hat{u})\Bigr)$$"| Koopman[Koopman\n Fitting]
    end
    Learning -->|"$$K_{\mathrm{Approx}}$$"| KR(Koopman\n Approximator) --> Prediction
    subgraph Prediction
        direction LR
        Init[Initial State]@{shape: lean-l} -->|"$$x_0$$"| Sum@{ shape: f-circ, label: "Junction" }
        Input[Input Sequence]@{shape: lean-l} -->|"$$\{u_k\}_{k=1}^N$$"| Sum
        Sum -->|"$$\Bigl(\hat{x}_0,\{u_k\}_{k=1}^N\Bigr)$$"| Scale2[Scaling]
        Scale2 -->|"$$\Bigl(\hat{x}_0,\{\hat{u}_k\}_{k=1}^N\Bigr)$$"| Lift2
        Scale2 -->|"$$\Bigl(\hat{x}_0,\{\hat{u}_k\}_{k=1}^N\Bigr)$$"| BilinearKoopman

        subgraph RegularKoopman[Linear or Bilinear]
            direction LR
            BilinearKoopman[Koopman\n Rollout]
        end

        subgraph LiftedKoopman[GeKo or KCF]
            direction LR
            Lift2[Lifitng] -->|"$$\Bigl(\varphi(\hat{x}),\{\psi(\hat{u}_k)\}_{k=1}^N\Bigr)$$"| KoopmanPredictor[Koopman\n Rollout]
        end
        
        BilinearKoopman -->|"$$\{\varphi(\hat{x})_k\}_{k=0}^{N+1}$$"| Junction@{ shape: f-circ, label: "Junction" }
        KoopmanPredictor -->|"$$\{\varphi(\hat{x})_k\}_{k=0}^{N+1}$$"| Junction --> Projection
        Projection -->|"$$\{\hat{x}_k\}_{k=0}^{N+1}$$"| Scaling2[Scaling]
        Scaling2 --> StateSequence["$$\{x_k\}_{k=0}^{N+1}$$"]
    end
```

## License

This project is licensed under the MIT license.

## Support

For any questions or issues related to this code, please contact the authors:

- Alex Erdin: [alex.erdin@tuhh.de](mailto:alex.erdin@tuhh.de)
- Lea Bold: [lea.bold@tu-ilmenau.de](mailto:lea.bold@tu-ilmenau.de)

We appreciate any feedback, bug reports, or suggestions for improvements.
