import gradio as gr
from movie_recommender import unique_genre,user_ids,gradio_recommend

with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Smart Movie Recommendation System (Flexible Mode)")

    user_id_dropdown = gr.Dropdown(choices=user_ids, label="Select User ID")

    # Optional Inputs
    movie_title_input = gr.Textbox(label="Enter Movie Title ")
    genre_dropdown = gr.Dropdown(choices=unique_genre, label="Select Genre (Optional)")
    mood_dropdown = gr.Dropdown(choices=["Neutral", "Happy", "Sad", "Excited", "Dark"], label="Select Mood (Optional)")

    num_recommendations_input = gr.Slider(1, 10, step=1, value=5, label="Number of Recommendations")

    output = gr.List(label="Recommended Movies")

    recommend_button = gr.Button("Get Recommendations")
    recommend_button.click(gradio_recommend, 
                           inputs=[user_id_dropdown, movie_title_input, genre_dropdown, mood_dropdown, num_recommendations_input], 
                           outputs=output)

demo.launch(share= True)