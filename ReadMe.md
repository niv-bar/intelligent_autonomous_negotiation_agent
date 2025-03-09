# 🚀 Setting Up Your Python Project in PyCharm

This guide provides step-by-step instructions for initializing your Python project in **PyCharm**, whether you have **Poetry installed** or not.

## 📌 Prerequisites
Ensure you have:
- **PyCharm** installed.
- **Python 3.10** installed.

---

## **Scenario 1: If You Have Poetry Installed**

1. **Open the Project in PyCharm**
   - Open PyCharm.
   - Click **"Open"** and select the project folder containing the `pyproject.toml` file.

2. **Install Dependencies**
   Open the terminal inside PyCharm and navigate to the project directory:
   ```bash
   cd /path/to/your/project
   poetry install
   ```
   This will create a virtual environment and install all dependencies listed in `pyproject.toml`.

3. **Set the Virtual Environment in PyCharm**
   - Go to **File** → **Settings** (or **PyCharm Preferences** on macOS).
   - Navigate to **Project: <your_project>** → **Python Interpreter**.
   - Click the gear icon ⚙️ and select **Add Interpreter** → **Add Local Interpreter**.
   - Choose **Poetry Environment** → **Use existing environment**.
   - Find the virtual environment path using:
     ```bash
     poetry env info --path
     ```
   - Select the environment and click **OK**.

---

## **Scenario 2: If You Don't Have Poetry Installed**

### **Option 1: Install Poetry (Recommended)**
1. Install Poetry using:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Verify installation:
   ```bash
   poetry --version
   ```
3. Follow **Scenario 1** steps to initialize the project.

### **Option 2: Use a Virtual Environment (Without Poetry)**
1. Open terminal and navigate to the project folder:
   ```bash
   cd /path/to/your/project
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
4. Install dependencies manually:
   ```bash
   pip install -r <(poetry export --without-hashes --format=requirements.txt)
   ```
   > If `poetry export` is not available, manually install dependencies listed in `pyproject.toml`.

5. Set up the virtual environment in PyCharm:
   - Go to **File** → **Settings** → **Project: <your_project>** → **Python Interpreter**.
   - Click **Add Interpreter** → **Add Local Interpreter**.
   - Select the virtual environment (`venv`) you created.
   - Click **OK**.

---

## **Installing All Packages in `pyproject.toml`**
To install all packages defined in `pyproject.toml`, run:
```bash
poetry install
```

If you want to install dependencies including `dev` dependencies, use:
```bash
poetry install --with dev
```

---

## 🎯 **You're All Set!**
Now you can start working on the Python project in PyCharm. 🚀 

Run the agent (terminal):
```bash
 python -m group14
```