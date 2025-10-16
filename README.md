# 2025-leaderboard-open
## Open version of 2025 DLAIE Latent Flow Matching Leaderboard 

This Leaderboard: https://2025-dlaie-leaderboard-open.streamlit.app/

Original Class Leaderboard: https://2025-dlaie-leaderboard.streamlit.app/



## 🚀 How to Contribute a Submission

We welcome new submissions to the leaderboard! Follow the steps below to ensure your entry is processed correctly and securely by our automated system.

### 1. Fork the Repository

First, create a **fork** of the main repository (`dlaieburner/2025-leaderboard-open`) to your personal GitHub account.

### 2. Prepare Your Submission File

1.  Create a **new branch** in your forked repository for your submission (e.g., `git checkout -b my-new-submission`).
2.  Create your Python submission file. Your file must contain the required model or function as specified in the contest rules.
3.  **Name Your File:** Your file name must be descriptive of your team or model, and it must be **unique** (e.g., `team-alpha.py` or `model-v3.py`).

### 3. Place the Submission File

Place your newly created Python file **ONLY** in the **`submissions/`** folder of your new branch.

> ⚠️ **Important:** Do not modify any files outside of your new submission file.

### 4. Create a Pull Request (PR)

1.  Commit your Python file to your new branch.
2.  Push the branch to your fork.
3.  Open a **Pull Request (PR)** from your branch back to the **`main`** branch of the original repository (`dlaieburner/2025-leaderboard-open`).

### 5. Automated Evaluation

Once your PR is opened:

1.  Our GitHub Action will automatically run a secure evaluation of your code.
2.  The full results, including scores and logs, will be posted **as a comment** in your Pull Request.
3.  You can iterate by pushing new changes to your PR branch; the system will automatically re-evaluate.

### 6. Final Status (Very Important)

* **We will NOT merge your original submission PR.** (The automated system securely extracts your code without merging your file into the repository).
* If your submission is successful, a maintainer will **manually approve the scores** and update the leaderboard.
* You will see your score reflected on the live Streamlit leaderboard shortly after approval.
