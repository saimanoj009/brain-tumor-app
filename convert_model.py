from tensorflow.keras.models import load_model

# Load existing model
model = load_model("brain_tumor_model.keras", compile=False)

# Save as .h5
model.save("brain_tumor_model.h5")

print("✅ Model converted successfully!")