import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel("Data/GTAttacksLogsFinal.xlsx")

# Columns containing the Hamming accuracy for each experiment
exp_cols = [
    "GPT+_TC-CAN_With_File_Same_Session_hamming",
    "GPT+_TC-CAN_Without_File_Same_Session_hamming",
    "GPT+_TC-CAN_Without_File_Different_Session_hamming",
    "GPT+_TC-CAN_With_File_Different_Session_hamming"
]

# Compute mean Hamming accuracy for the bar chart
means = df[exp_cols].mean()

plt.figure(figsize=(10,6))
plt.bar(exp_cols, means)

plt.title("Average Hamming Accuracy per Experiment")
plt.ylabel("Mean Hamming Accuracy")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
