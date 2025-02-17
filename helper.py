from ultralytics import YOLO
import streamlit as st
import cv2
import pafy
import pickle
import settings

# Set pafy backend to avoid youtube_dl issues
# pafy.set_backend("internal")

# Load model
with open('yolo.pkl', 'rb') as file:
    model1 = pickle.load(file)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    model = YOLO('yoloooo.pt')
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = display_tracker == 'Yes'
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """ Display detected objects on a video frame using the YOLOv8 model. """
    image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.track(image, conf=conf, persist=True, tracker=tracker) if is_display_tracking else model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)

def play_youtube_video(conf, model):
    """ Plays a YouTube video and detects objects in real-time. """
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Trash'):
        try:
            video = pafy.new(source_youtube)
            best = video.getbest(preftype="mp4")
            vid_cap = cv2.VideoCapture(best.url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error loading video: {e}")

def play_webcam(conf, model):
    """ Plays a webcam stream and detects objects in real-time. """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Trash'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error loading video: {e}")

def play_stored_video(conf, model):
    """ Plays a stored video and detects objects in real-time. """
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Video Trash'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error loading video: {e}")
