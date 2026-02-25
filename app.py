import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("🌳 ID3 Decision Tree - Play Tennis Prediction")

st.write("This app builds a Decision Tree using the ID3 algorithm and predicts whether to Play Tennis.")

# -----------------------------
# Step 1: Dataset
# -----------------------------
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                'Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal',
                 'Normal','High','Normal','High','Normal','High',
                 'Normal','High'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No',
                    'Yes','No','Yes','Yes','Yes','Yes',
                    'Yes','No']
})

st.subheader("📊 Dataset")
st.dataframe(data)

# -----------------------------
# Step 2: Entropy Function
# -----------------------------
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    ent = 0
    for count in counts:
        p = count / len(col)
        ent -= p * math.log2(p)
    return ent

# -----------------------------
# Step 3: Information Gain
# -----------------------------
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attribute], return_counts=True)

    weighted_entropy = 0
    for i in range(len(values)):
        subset = df[df[attribute] == values[i]]
        weighted_entropy += (counts[i]/len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# -----------------------------
# Step 4: ID3 Algorithm
# -----------------------------
def id3(df, target, attributes):

    if len(np.unique(df[target])) == 1:
        return df[target].iloc[0]

    if len(attributes) == 0:
        return df[target].mode()[0]

    gains = [information_gain(df, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]

    tree = {best_attr: {}}

    for value in np.unique(df[best_attr]):
        subset = df[df[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = id3(subset, target, remaining_attrs)

    return tree

# -----------------------------
# Step 5: Build Decision Tree
# -----------------------------
attributes = list(data.columns)
attributes.remove('PlayTennis')

decision_tree = id3(data, 'PlayTennis', attributes)

st.subheader("🌲 Generated Decision Tree")
st.json(decision_tree)

# -----------------------------
# Step 6: Prediction Function
# -----------------------------
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    value = sample[attr]

    if value in tree[attr]:
        return predict(tree[attr][value], sample)
    else:
        return "Unknown"

# -----------------------------
# Step 7: User Input
# -----------------------------
st.sidebar.header("🔎 Enter Conditions")

outlook = st.sidebar.selectbox("Outlook", data["Outlook"].unique())
humidity = st.sidebar.selectbox("Humidity", data["Humidity"].unique())

sample = {'Outlook': outlook, 'Humidity': humidity}

if st.sidebar.button("Predict"):
    result = predict(decision_tree, sample)

    st.subheader("🎯 Prediction Result")
    st.write("Input:", sample)
    
    if result == "Yes":
        st.success("Play Tennis: YES 🎾")
    elif result == "No":
        st.error("Play Tennis: NO ❌")
    else:
        st.warning("Prediction: Unknown")
