
from pathlib import Path
import pandas as pd

Path("data").mkdir(exist_ok=True)

data = [
    ("The university published its official admission schedule.", "REAL"),
    ("The weather department released its forecast.", "REAL"),
    ("The city council approved a new public library.", "REAL"),
    ("The railway department announced a timetable update.", "REAL"),
    ("The college opened online registration.", "REAL"),
    ("The health department issued a public advisory.", "REAL"),
    ("A viral post claims humans can live without water for years.", "FAKE"),
    ("A message claims every student will receive a free luxury car.", "FAKE"),
    ("A viral post claims the moon disappeared last night.", "FAKE"),
    ("An online message promises instant wealth from a secret trick.", "FAKE"),
    ("A post claims one food cures every disease overnight.", "FAKE"),
    ("A viral message claims ordinary phones can control the weather.", "FAKE"),
]

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/news.csv", index=False)

print("Dataset created successfully!")
print(df)
