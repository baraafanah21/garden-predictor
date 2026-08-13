import pandas as pd

REQUIRED_COLUMNS = ["soil_moisture", "last_watered", "plant_type", "needs_water", "x", "y"]


def loadTrainingData(filePath):
    df = pd.read_excel(filePath)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column in Excel file: {col}")

    X = df[["soil_moisture", "last_watered", "plant_type"]].values.tolist()
    y = df["needs_water"].values.tolist()

    return X, y, df


def normalizeFeatures(X):
    normalized = []

    for row in X:
        soil_moisture = row[0] / 100.0
        last_watered = row[1] / 48.0
        plant_type = row[2] / 2.0

        normalized.append([soil_moisture, last_watered, plant_type])

    return normalized


def split_data(X, y, train_ratio):
    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Train ratio leaves no data for training or testing.")

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


def preparePlants(df, model):
    plants = []

    for i in range(len(df)):
        soil_moisture = df.loc[i, "soil_moisture"]
        last_watered = df.loc[i, "last_watered"]
        plant_type = df.loc[i, "plant_type"]
        x = df.loc[i, "x"]
        y = df.loc[i, "y"]

        norm = normalizeFeatures([[soil_moisture, last_watered, plant_type]])[0]
        predicted = model.predictSingle(norm)

        plant = {
            "id": i + 1,
            "soil_moisture": soil_moisture,
            "last_watered": last_watered,
            "plant_type": plant_type,
            "x": x,
            "y": y,
            "predicted": predicted
        }
        plants.append(plant)

    return plants


def countPredicted(plants):
    return sum(1 for p in plants if p["predicted"] == 1)


def getNeedy(plants):
    # Returns only plants the model predicts need water.
    return [p for p in plants if p["predicted"] == 1]

