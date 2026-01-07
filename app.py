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
    try:
        # STEP 2: Enhanced SerpApi extraction
        lens_response = requests.get("https://serpapi.com/search", params=lens_params).json()
        visual_matches = lens_response.get("visual_matches", [])[:8] # Get more results

        # Capture the image URL (thumbnail) along with title and link
        context_data = [
            {
                "title": m.get("title"), 
                "link": m.get("link"), 
                "thumbnail": m.get("thumbnail"), # <--- KEY: Capture the image
                "price": m.get("price", {}).get("extracted_value", "Unknown")
            } for m in visual_matches
        ]

        # STEP 3: Updated Prompt to include images
        system_prompt = "You are an elite AI Fashion Stylist. Respond ONLY in valid JSON."
        user_prompt = f"""
        Using this data: {context_data}

        TASK:
        1. Identify the 'main_item' (the best match).
        2. Create a list of 'similar_items' (3-4 items) including their names, prices, and THUMBNAIL URLs.

        JSON STRUCTURE:
        {{
            "main_item": {{
                "name": "", "brand": "", "price": "", "buy_url": "", "image_url": "" 
            }},
            "similar_items": [
                {{ "name": "", "price": "", "buy_url": "", "image_url": "" }}
            ],
            "style_vibe": "",
            "suggestions": []
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
