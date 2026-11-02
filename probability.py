"""
Step 4 - Probability: events, conditional probability, independence,
total probability and Bayes' theorem on the bike data.

Event definitions (on the full clean data):
    High  = rentals above the overall median
    Rain  = rainfall > 0 mm in that hour
    Rush  = hour is 7-9 am or 5-7 pm (17-19)
"""
import numpy as np
import pandas as pd


def run_probability(df):
    d = df.copy()
    d["High"] = d["Count"] > d["Count"].median()
    d["RainEvt"] = d["Rain"] > 0
    d["Rush"] = d["Hour"].isin([7, 8, 9, 17, 18, 19])
    n = len(d)

    p_high = d["High"].mean()
    p_rain = d["RainEvt"].mean()
    p_rush = d["Rush"].mean()
    res = {"P(High)": p_high, "P(Rain)": p_rain, "P(Rush)": p_rush}

    # ---- Conditional probability: P(A|B) = P(A and B) / P(B)
    p_high_rain = d.loc[d["RainEvt"], "High"].mean()
    p_high_norain = d.loc[~d["RainEvt"], "High"].mean()
    p_high_rush = d.loc[d["Rush"], "High"].mean()
    p_high_nonrush = d.loc[~d["Rush"], "High"].mean()
    res.update({"P(High|Rain)": p_high_rain, "P(High|NoRain)": p_high_norain,
                "P(High|Rush)": p_high_rush, "P(High|NotRush)": p_high_nonrush})
    for s in ["Spring", "Summer", "Autumn", "Winter"]:
        res[f"P(High|{s})"] = d.loc[d["Season"] == s, "High"].mean()

    # ---- Independence check: if independent, P(High and Rain) = P(High) * P(Rain)
    p_joint = (d["High"] & d["RainEvt"]).mean()
    res["P(High and Rain)"] = p_joint
    res["P(High)*P(Rain)"] = p_high * p_rain

    # ---- Law of total probability: P(High) = sum over seasons of P(High|S) P(S)
    total = sum(d.loc[d["Season"] == s, "High"].mean() * (d["Season"] == s).mean()
                for s in d["Season"].unique())
    res["P(High) via total probability"] = total

    # ---- Bayes' theorem: P(Rain|High) = P(High|Rain) P(Rain) / P(High)
    bayes = p_high_rain * p_rain / p_high
    direct = d.loc[d["High"], "RainEvt"].mean()
    res["P(Rain|High) by Bayes"] = bayes
    res["P(Rain|High) direct count"] = direct

    # A more useful Bayes question: P(Rush | High)
    res["P(Rush|High) by Bayes"] = p_high_rush * p_rush / p_high

    print("\n=== Probability results ===")
    for k, v in res.items():
        print(f"{k:34s} {v:.4f}")
    print(f"\nHigh demand is {p_high_norain/p_high_rain:.1f}x as likely without rain as with rain.")
    print("Independence of High and Rain? P(High and Rain)="
          f"{p_joint:.4f} vs P(High)P(Rain)={p_high*p_rain:.4f} -> NOT independent"
          if abs(p_joint - p_high * p_rain) > 0.005 else "-> roughly independent")
    return {k: float(v) for k, v in res.items()}
