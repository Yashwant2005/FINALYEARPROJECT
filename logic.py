def refine_prediction(disease, confidence, temp, humidity):

    status = "Normal"

    # humidity rule
    if humidity > 80 and "Blight" in disease:
        confidence += 0.05
        status = "High Risk due to humidity"

    # severity
    if confidence > 0.85:
        severity = "Severe"
    elif confidence > 0.65:
        severity = "Moderate"
    else:
        severity = "Early Stage"

    return severity, status, round(confidence, 2)