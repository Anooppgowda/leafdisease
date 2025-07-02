import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow Model prediction 


def model_prediction(test_image):
    cnn = tf.keras.models.load_model("C:/Users/VISHNU.S/Desktop/aicte_project/leaf_disease/trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions) #Return index of max element
    return result_index

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode == "Home"):
    st.header("Plant Disease Recognition System")
    image_path = "C:/Users/VISHNU.S/Desktop/aicte_project/home.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
                
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, 
    and our system will analyze it to detect any signs of diseases. Together, 
    let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant
    with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify 
    potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for 
    accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience 
    the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                """)
    
#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves 
                which is categorized into 38 different classes.The total dataset is divided into 
                80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name =  ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                       'Apple___healthy','Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy', 'Grape___Black_rot', 
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
               'Strawberry___Leaf_scorch','Strawberry___healthy', 'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
               'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        
        treatment_dict = {
    'Apple___Apple_scab': 'Use fungicide sprays and resistant varieties.',
    'Apple___Black_rot': 'Remove infected branches and apply fungicide.',
    'Apple___Cedar_apple_rust': 'Apply fungicides early in the season.',
    'Apple___healthy': 'No treatment needed.',
    'Blueberry___healthy': 'No treatment needed.',
    'Cherry_(including_sour)___Powdery_mildew': 'Use sulfur-based fungicide and prune properly.',
    'Cherry_(including_sour)___healthy': 'No treatment needed.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids and rotate crops.',
    'Corn_(maize)___Common_rust_': 'Use fungicide if infection is severe.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Plant resistant hybrids, apply fungicide if needed.',
    'Corn_(maize)___healthy': 'No treatment needed.',
    'Grape___Black_rot': 'Prune infected parts and apply fungicide.',
    'Grape___Esca_(Black_Measles)': 'Remove infected vines, no chemical control available.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides and improve air circulation.',
    'Grape___healthy': 'No treatment needed.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure, remove infected trees.',
    'Peach___Bacterial_spot': 'Use disease-free seeds and apply bactericides.',
    'Peach___healthy': 'No treatment needed.',
    'Pepper,_bell___Bacterial_spot': 'Use copper-based sprays.',
    'Pepper,_bell___healthy': 'No treatment needed.',
    'Potato___Early_blight': 'Use certified seeds and fungicides.',
    'Potato___Late_blight': 'Apply systemic fungicides.',
    'Potato___healthy': 'No treatment needed.',
    'Raspberry___healthy': 'No treatment needed.',
    'Soybean___healthy': 'No treatment needed.',
    'Squash___Powdery_mildew': 'Apply sulfur-based fungicide.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves, avoid overhead watering.',
    'Strawberry___healthy': 'No treatment needed.',
    'Tomato___Bacterial_spot': 'Use copper sprays and resistant varieties.',
    'Tomato___Early_blight': 'Apply fungicides and practice crop rotation.',
    'Tomato___Late_blight': 'Use resistant varieties, apply fungicide.',
    'Tomato___Leaf_Mold': 'Improve ventilation and apply fungicide.',
    'Tomato___Septoria_leaf_spot': 'Use drip irrigation and apply fungicide.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray miticide or insecticidal soap.',
    'Tomato___Target_Spot': 'Apply recommended fungicides.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies, remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected plants, sanitize tools.',
    'Tomato___healthy': 'No treatment needed.'
}
        
        st.success("Model is Predicting it's a {} disease ".format(class_name[result_index]))
        
        treatment = treatment_dict.get(class_name[result_index], "No treatment information available.")
        st.info(f"**Treatment Recommendation:** {treatment}")

