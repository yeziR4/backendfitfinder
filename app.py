import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
from groq import Groq
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. Cloudinary Config
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

# 2. Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SERP_API_KEY = os.getenv("SERP_API_KEY")

@app.route('/fitfinder', methods=['POST'])
def fit_finder():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['file']
    
    try:
        # STEP 1: Upload to Cloudinary & Get Public URL
        upload_result = cloudinary.uploader.upload(file)
        image_url = upload_result.get("secure_url")

        # STEP 2: Feed URL into SerpApi (Google Lens)
        lens_params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": SERP_API_KEY
        }
        lens_response = requests.get("https://serpapi.com/search", params=lens_params).json()
        
        # Extract visual matches (titles and links)
        visual_matches = lens_response.get("visual_matches", [])[:5]
        context_data = [{"title": m.get("title"), "link": m.get("link")} for m in visual_matches]

        # STEP 3: Llama 3.3 Stylist Reasoning
        system_prompt = "You are an elite AI Fashion Stylist. Respond ONLY in a clean JSON format."
        user_prompt = f"""
        Analyze these fashion items found in an image: {context_data}.
        
        1. Identify the exact Brand and Product Name for the main item.
        2. Provide a 'Market Price' estimate.
        3. Give a direct 'Buy Link' (use the most relevant link from the data).
        4. Suggest 2 other clothing items that match this 'Aesthetic'.
        
        Return JSON structure:
        {{
            "main_item": {{"name": "", "brand": "", "price": "", "buy_url": ""}},
            "style_vibe": "",
            "suggestions": ["item1", "item2"]
        }}
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        return completion.choices[0].message.content

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
