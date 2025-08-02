**Authors**:
Zachary Holland & Devlin Bridges (Group 2)


<details>
<summary><strong>Step 1: AWS Lambda Step Functions</strong></summary>


## Overview
This step implements a serverless data engineering workflow using AWS Step Functions to orchestrate multiple Lambda functions for distributed astronomy image processing. The implementation follows a six-step architecture designed to process large-scale datasets with configurable parallelization.

## Authors
**Zachary Holland** & **Devlin Bridges** (Group 2)

## Architecture Components

### 1. **Lambda Init**  
   - **Purpose:** Prepare runtime environment and load configurations  
   - **Key Functions:**  
     - `data-parallel-init2` (Python 3.13, 128 MB, 60 s timeout)  
     - `data-parallel-init-fmi` (Python 3.13, 128 MB, 63 s timeout)  
     - `init` (Python 3.13, 128 MB, 3 s timeout)  
     - `fmi_init` (Python 3.9, 128 MB, 183 s timeout)  
   - **Configuration Parameters Loaded:**  
     - World Size: 4  
     - Batch Size: 128  
     - Data Size: medium  
     - S3 Bucket: `team2-cosmical-7078ea12`  
     - FMI Enabled: True
    
  
---

### 2. **Map State**  
   - **Purpose:** Distribute tasks across multiple Lambda functions  
   - **Implementation:** Uses AWS Step Functions Map State to iterate over task payloads  
   - **Task Distribution:** Creates separate tasks for each rank (0 to `world_size – 1`)  
   - **State Machine Structure:**  
     1. **Lambda Invoke**: Task  
     2. **Distributed**: Map (with configurable `MaxConcurrency`)  
     3. **Summarize**: Task  
---

### 3. Extract and Invoke

**Purpose**: Handle data allocation and inference execution

**Inference Functions**:

* `inference` (Python 3.13, 128MB, 60s timeout)
* `data-parallel-init-inf` (Python 3.13, 128MB, 3s timeout)

**Each Lambda Receives**:

* Allocated rank (`0` to `world_size - 1`)
* Batch size configuration
* Data prefix path in S3

---

### 4. Lambda Invoke FMI

**Purpose**: Provide distributed task synchronization

**FMI Configuration**:

* **FMI Enabled**: `True`
* **Rendezvous Endpoint**: `rendezvous.uva-ds5110.com:10000`
* **World Size**: 2–4 (distributed tasks)

**Key Function**: `data-parallel-init-fmi` (Python 3.13, 128MB, 63s timeout)

---

### 5. End State

**Purpose**: Process and store final results

**Result Processing Functions**:

* `summarize` (Python 3.13, 128MB, 60s timeout)
* `resultSummary` (Python 3.12, 128MB, 150s timeout)

**Output**: Combined results uploaded to S3:

```
results/combined_data.json
```

---

### 6. Performance Measurement

**Test Configurations**: 20 different combinations varying:

* World Size: `1, 2, 3, 4`
* Batch Size: `8, 16, 32, 64, 128`

**Metrics Collected**:

* Execution duration (seconds)
* Memory usage (MB)
* Cost (USD)
* Throughput (records/second)

#### Performance Results Summary

**Average Performance by World Size**:

| World Size | Avg Time (s) | Throughput (records/sec) |
| ---------- | ------------ | ------------------------ |
| 1          | 5.30         | 9.67                     |
| 2          | 5.11         | 19.44                    |
| 3          | 6.54         | 24.02                    |
| 4          | 7.55         | 28.07                    |

**Best Performance Configuration**:

* **World Size**: 4
* **Batch Size**: 128
* **Throughput**: 71.42 records/second

**Key Findings**:

* \~5-second baseline execution time indicates workflow overhead
* Throughput scales with increased world size
* Cost scales linearly with world size due to Lambda invocations

---

## IAM Configuration

* **Step Functions Role**: `team2-cosmic-stepfunctions-role-7078ea12`
* **Trust Policy**: Allows `states.amazonaws.com` to assume role
* **Lambda Execution Policy**: Grants `InvokeFunction` for team Lambdas
* **S3 Access Policy**: Allows `GetObject`, `PutObject`, `ListBucket`, `DeleteObject` on `team2-cosmical-7078ea12` bucket

---

## Implementation Details

* **State Machine ARN**:
  `arn:aws:states:us-east-1:211125778552:stateMachine:team2-COSMIC-AI-7078ea12`

* **Sample JSON Payload**:

```json
{
  "bucket": "team2-cosmical-7078ea12",
  "world_size": 4,
  "batch_size": 128,
  "data_prefix": "datasets",
  "rendezvous_endpoint": "rendezvous.uva-ds5110.com:10000",
  "fmi_enabled": true,
  "data_map": {
    "0": null,
    "1": null,
    "2": null,
    "3": null
  }
}
```

---

## Challenges Encountered

* **Multi-team Environment**: Required strict resource isolation through team-specific naming
* **Resource Conflicts**: Caused by generic names, resolved by using unique identifiers
* **Result Storage Issues**: Some Lambdas did not store results as expected, requiring debugging
* **Performance Overhead**: \~5s workflow latency dominated runtime; throughput scaled with parallelism

---

## Key Achievements

* Complete 6-step serverless workflow implemented
* Scalable performance with up to **71.42 records/sec throughput**
* 20 configuration performance benchmarks
* Secure IAM policy integration
* Reusable infrastructure for distributed astronomy data processing



</details>



<details>
<summary><strong>Step 2: Rendezvous Server</strong></summary>

## Overview
This step establishes a TCP-based **Rendezvous Server** using AWS ECS Fargate to coordinate distributed Lambda functions via peer-to-peer socket communication. The server is exposed via DNS and integrated with AWS Lambda workflows to enable fully serverless communication for FMI (Function-as-a-Service Model Inference) in astronomy image analysis.

## Authors
**Zachary Holland** & **Devlin Bridges** (Group 2)

---

## Architecture Components

### 1. **ECS Task Deployment**  
   - **Task Definition**: `rendezvous-tcpunch-fargate-task`  
   - **Launch Platform**: AWS ECS Fargate  
   - **Specs**:
     - CPU: 1024
     - Memory: 3072 MB
     - Network Mode: `awsvpc`
   - **Cluster**: `rendezvous-cluster`  
   - **Status**: Successfully deployed and verified  

---

### 2. **Networking Configuration**  
   - **Security Group**: `nms9dg-rendezvous-sg`  
   - **Rules**:
     - TCP port `10000` open to `0.0.0.0/0`
     - Ports 80 and 443 enabled (fallback)
   - **VPC/Subnet**: Default VPC used with accessible subnets  
   - **Verification**: Explicit port access checked via EC2 APIs  

---

### 3. **Public Endpoint Provisioning**  
   - **Public IP**: `54.146.211.10`  
   - **DNS Record**: `rendezvous.uva-ds5110.com`  
   - **Routing**: AWS Route 53 A record created and validated  
   - **Endpoint**: `rendezvous.uva-ds5110.com:10000`

---

### 4. **Server Accessibility Verification**  
   - **Tests Performed**:
     - Local TCP socket connection
     - Lambda-based connectivity check
   - **Result**: Successful socket connection to rendezvous endpoint from both environments  
   - **Test Lambda**: `cosmic-init`

---

### 5. **Lambda Inter-Communication**

**Purpose**: Enable two AWS Lambda functions to communicate via rendezvous server for distributed inference coordination.

**Test Functions**:

- `cosmic-init` (initiator)
- `cosmic-executor` (responder)

**Outcome**:

- Inter-Lambda communication succeeded
- FMI coordination verified

---

### 6. **Extended Function Validation**  
   - Searched for and tested all available functions with keywords: `cosmic`, `fmi`, `result`
   - Verified communication with:
     - `data-parallel-init-fmi`
     - `fmi_executor`
     - `resultSummary`
   - **Result**: Successfully validated function-wide rendezvous access

---

## FMI Integration Preparation

### 7. **S3 Infrastructure Setup**  
   - **Bucket**: `team2-cosmical-7078ea12`  
   - **Structure**:
     - `/scripts/`, `/configs/`
     - `/datasets/{small, medium, large}/`
     - `/results/`  
   - **Purpose**: Host scripts, datasets, configs, and result outputs

---

### 8. **Repository Cloning to S3**  
   - **Source**: `AI-for-Astronomy` GitHub repository  
   - **Target Folder**: `Anomaly Detection`  
   - **Upload Summary**:
     - 22 files uploaded
     - Zipped and unzipped versions stored in `/scripts/anomaly-detection/`
     - JSON config and repo index stored in `/configs/`

---

### 9. **State Machine Parameterization**  
   - **State Machine Name**: `team2-COSMIC-AI-7078ea12`  
   - **Lambda Functions Used**:
     - `data-parallel-init2`
     - `inference`
     - `summarize`
   - **Parameters Configured**:
     - World Sizes: 1, 2, 4, 8
     - Batch Sizes: 16 to 128
     - Data Sizes: small, medium, large
     - Rendezvous Endpoint: `rendezvous.uva-ds5110.com:10000`
   - **Role Used**: `team2-cosmic-stepfunctions-role-7078ea12`

---

### 10. **Test Executions Launched**

**Test Matrix**:
10 scenarios launched varying:

- **World Sizes**: 1, 2, 4, 8  
- **Batch Sizes**: 16, 32, 64, 128  
- **Data Sizes**: small, medium, large  

**Status**: All executions launched via Step Functions, FMI-enabled

---

## IAM Configuration

- **Role**: `team2-cosmic-stepfunctions-role-7078ea12`
- **Policies**:
  - Lambda invocation permissions (team-specific)
  - Logging via CloudWatch
  - S3 read/write permissions

---

## Implementation Details

- **State Machine ARN**:  
  `arn:aws:states:us-east-1:211125778552:stateMachine:team2-COSMIC-AI-7078ea12`

- **Sample Payload**:

```json
{
  "world_size": 4,
  "batch_size": 128,
  "data_size": "medium",
  "S3_object_name": "batch_4.json",
  "bucket": "team2-cosmical-7078ea12",
  "rendezvous_endpoint": "rendezvous.uva-ds5110.com:10000",
  "unique_id": "7078ea12",
  "result_path": "results/world_4"
}
```

---

## Challenges Encountered

- **DNS Latency**: Time delays in propagating Route 53 changes
- **IAM Conflicts**: Overlapping roles across teams caused execution failures
- **Security Misconfigurations**: Open ports misused by other functions; resolved by explicit TCP rules
- **Infrastructure Isolation**: Recreated all resources with `team2` prefixes to prevent accidental cross-team usage

---

## Key Achievements

- Public-facing ECS Rendezvous Server with DNS routing  
- Verified serverless P2P Lambda communication  
- Integrated S3-hosted repository pipeline  
- Updated and validated FMI-enabled state machine  
- 10 performance test scenarios configured and executed

</details>



<details>
<summary><strong>Step 3: Astronomy Inference</strong></summary>

## Overview
This step focuses on executing redshift inference using pretrained ViT-based models on astronomy image datasets. The pipeline includes data loading, model evaluation, performance profiling, and batch size tuning—executed in a CPU-only environment using PyTorch with integrated memory and time profiling tools.

## Authors
**Zachary Holland** & **Devlin Bridges** (Group 2)

---

## Inference Pipeline Components

### 1. **Repository Setup**  
   - **Cloning Source**:  
     ```bash
     git clone https://github.com/UVA-MLSys/AI-for-Astronomy.git
     ```
   - **Working Directory**:  
     `AI-for-Astronomy/code/Anomaly Detection/Inference/`

   - **Environment Adjustments**:
     - Local paths modified in `inference.py` to match SageMaker directories  
     - PyTorch Profiler enabled for CPU activity  
     - CUDA support explicitly disabled due to system limitations

---

### 2. **Inference Logic**

**Key Functions**:
- `load_data()`: Loads the input `.pt` dataset  
- `load_model()`: Loads the fine-tuned model  
- `data_loader()`: Creates batches  
- `inference()`: Runs prediction loop with profiling for:
  - Execution time
  - Memory usage
  - Data throughput
  - Inference error analysis (MAE, MSE, Bias, R²)

**Execution Command**:
```bash
python inference.py --batch_size 32 --device cpu
```

---

### 3. Inference Results (Baseline)

**Device**: CPU  
**Batch Size**: 32  
**Runtime Metrics**:
- Total Execution Time: 131.84s  
- Average Time per Batch: 168.81 ms  
- Throughput: 31,118,882 bits/sec

**Prediction Metrics**:
- MAE: 0.0134  
- MSE: 0.00038  
- Bias: 0.00292  
- R² Score: 0.968  

---

### 4. Batch Size Benchmarking

**Tested Configurations**:  
Batch sizes: `1, 2, 8, 16, 32, 64`

**Summary Table**:

| Batch Size | Exec Time (s) | Throughput (bps) | Avg Time/Batch (ms) | R² Score | MAE     |
|------------|----------------|------------------|----------------------|----------|---------|
| 1          | 85.72          | 2.40M            | 68.41                | 0.9747   | 0.01252 |
| 2          | 49.73          | 4.14M            | 79.31                | 0.9747   | 0.01252 |
| 8          | 18.27          | 11.26M           | 116.40               | 0.9747   | 0.01252 |
| 16         | 11.67          | 17.63M           | 147.67               | 0.9747   | 0.01252 |
| 32         | 131.84         | 31.12M           | 168.81               | 0.9684   | 0.01337 |
| 64         | 6.73           | 30.56M           | 336.52               | 0.9747   | 0.01252 |

**Key Findings**:
- Fastest Execution: **Batch 64** (6.73s)  
- Best Accuracy: **Batch 1** (R² = 0.9747)  
- Most Efficient per Batch: **Batch 1** (68.41 ms)  
- Highest Throughput: **Batch 32** (31.12M bps)


**Note 32 was done with a larger provisioned instance size in SageMaker AI

<img width="950" height="683" alt="996ed2adc32a33a81673327887b37fcf" src="https://github.com/user-attachments/assets/4b1aa3da-e42a-4368-abe8-8363c773f984" />


---

### 5. System Configuration

| Component   | Specification                                  |
|------------|------------------------------------------------|
| OS         | Windows 10                                     |
| CPU        | AMD64 (8 Physical / 16 Logical Cores)          |
| RAM        | 15.34 GB (10.71 GB available)                  |
| GPU        | Not available                                  |
| Execution  | CPU-only                                       |

---



### 6. Extended Batch Experiments (Series 2)

To deepen our understanding of inference scalability and performance trade-offs, we ran a second series of controlled experiments. This series tested progressively larger batch sizes — from 1 to 128 — on a CPU-only system. Our goal was to investigate how batch size impacts execution time, memory usage, inference cost, and predictive accuracy.

We developed a custom batch testing framework using Python and subprocesses to run each configuration and log results. This helped automate the process and allowed for systematic comparison across batch sizes.

#### Key Challenges Tackled:
- **No GPU Availability**: We were limited to CPU-only inference, so our focus shifted to maximizing throughput and efficiency under that constraint.
- **Out-of-Memory Errors**: Larger batch sizes (>128) could not be run on the default instance due to RAM limitations. We capped testing at 128 to avoid instability.
- **Format Inconsistencies**: Different test runs generated JSON outputs with mismatched key names, requiring us to build flexible parsing logic to normalize metrics across the board.

By observing both execution-level metrics (e.g., total time, memory usage, cost) and prediction metrics (e.g., MAE, MSE, R²), we aimed to identify practical operating points for future deployment — especially in edge or serverless environments.

**Batch Sizes Tested**: `1, 2, 4, 8, 16, 32, 64, 128`

**Execution Results**:

| Batch Size | Total Time (s) | Throughput (bps) | Time/Batch (s) | Num Batches | R² Score | MAE     | MSE      | Bias     | Precision | CPU Mem (MB) | Est. Cost (USD) |
|------------|----------------|------------------|----------------|-------------|----------|---------|----------|----------|-----------|---------------|-----------------|
| 1          | 112.79         | 1.82M            | 0.09001        | 1253        | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 24149.78      | 0.011630        |
| 2          | 60.77          | 3.38M            | 0.09693        | 627         | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25377.38      | 0.006266        |
| 4          | 36.25          | 5.67M            | 0.11546        | 314         | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25226.69      | 0.003738        |
| 8          | 20.96          | 9.82M            | 0.13348        | 157         | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25143.06      | 0.002161        |
| 16         | 13.98          | 14.71M           | 0.17699        | 79          | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25126.06      | 0.001442        |
| 32         | 10.32          | 19.93M           | 0.25806        | 40          | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25182.75      | 0.001064        |
| 64         | 8.43           | 24.41M           | 0.42130        | 20          | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25175.41      | 0.000869        |
| 128        | 7.21           | 28.53M           | 0.72094        | 10          | 0.974674 | 0.01252 | 0.000297 | 0.002024 | 0.011136  | 25215.61      | 0.000743        |



<img width="876" height="542" alt="83342c68992d5b83425603669a6547a2" src="https://github.com/user-attachments/assets/c4e9bb46-6e7e-420d-8fb5-e1e1fb5ec005" />



---

**Key Insights:**
- **Accuracy was stable**: R² remained constant at 0.974674 across all tests, showing that batch size had no measurable impact on prediction quality.
- **Execution speed improved** with larger batches. Batch size 128 was over **15× faster** than size 1.
- **Cost per inference** dropped drastically with size 128 — as low as ~$0.00074.
- **Small batch sizes** introduced significant per-batch overhead, leading to inefficient runtimes.
- **Memory usage was high but stable**, with most runs using 24–25 GB of RAM.

**Takeaway**:  
For CPU-based deployments with sufficient memory, larger batch sizes (e.g. 64–128) are optimal. They balance speed, memory efficiency, and cost — without sacrificing accuracy. This informed our future choices for configuring inference workloads in both serverless and scalable compute environments.
</details>
