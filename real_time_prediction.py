import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado
model = tf.keras.models.load_model("my_model.h5")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplicar filtro de mediana para reducir ruido
        blurred = cv2.medianBlur(gray, 5)
        
        # 3. Aplicar umbral adaptativo
        threshed = cv2.adaptiveThreshold(blurred, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # 4. Redimensionar a 28x28 (tamaño MNIST)
        resized = cv2.resize(threshed, (28, 28))
        
        # 5. Preparar la imagen para el modelo
        input_img = resized.reshape(1, 28, 28, 1).astype('float32') / 255
        
        # 6. Hacer la predicción
        prediction = model.predict(input_img, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 7. Mostrar solo la predicción en la esquina superior izquierda
        text = f"{predicted_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar el frame principal
        cv2.imshow('Reconocimiento de Digitos', frame)
        
        # Mostrar vista previa del procesamiento (opcional)
        processed_view = cv2.resize(resized, (280, 280), 
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Vista Procesada', processed_view)

        # Salir con 'q' o ESC
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()