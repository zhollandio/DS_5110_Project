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

<details>

