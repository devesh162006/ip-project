import cv2
import os
import sys
import logging
import tensorflow

try:
    
    from keras.models import load_model, Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    KERAS_BACKEND = 'tensorflow'
    logging.info("Using TensorFlow Keras backend")
except Exception as e_tf:
    try:
        
        from keras.models import load_model, Sequential
        from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
        KERAS_BACKEND = 'keras'
        logging.info("Using standalone Keras backend")
    except Exception as e_keras:
        logging.exception("Failed to import tensorflow.keras and standalone keras")
        raise ImportError("Please install TensorFlow (tensorflow) or Keras (keras) to use this script") from e_keras
import numpy as np
from pygame import mixer
import time
import argparse
import logging


ALARM_PATH = r"C:\Users\deves\Downloads\Alert.wav"

sound = None



path = os.getcwd()

def _load_cascade(local_subpath, fallback_name):
    
    
    candidates = [
        os.path.join('haar cascade files', local_subpath),
        local_subpath,
        os.path.join(cv2.data.haarcascades, fallback_name)
    ]
    
    path_found = None
    for p in candidates:
        if os.path.exists(p):
            path_found = p
            break
            
    if path_found is None:
        
        path_found = os.path.join(cv2.data.haarcascades, fallback_name)

    cc = cv2.CascadeClassifier(path_found)
    if cc.empty():
        logging.error(f"Failed to load cascade: {local_subpath} or {fallback_name} (checked paths: {candidates})")
    else:
        logging.info(f"Loaded cascade {local_subpath} -> {fallback_name} from {path_found}")
    return cc


face = _load_cascade('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt.xml')

leye = _load_cascade('haarcascade_lefteye_2splits.xml', 'haarcascade_eye.xml')
reye = _load_cascade('haarcascade_righteye_2splits.xml', 'haarcascade_eye.xml')

lbl='Open'

MODEL_PATH = os.path.join('models','cnncat2.h5')
CAMERA_INDEX = 0

ALERT_THRESHOLD = 15      
ALERT_COOLDOWN = 10      
SCORE_INCREMENT = 1        
SCORE_DECREMENT = 1        

FPS_SMOOTHING = 0.9

LOG_PATH = os.path.join(path, 'drowsiness.log')


parser = argparse.ArgumentParser(description='Drowsiness detection')
parser.add_argument('--model', '-m', help='Path to model .h5', default=None)
parser.add_argument('--alarm', help='Path to alarm wav', default=None)
parser.add_argument('--camera', type=int, help='Camera index', default=None)
parser.add_argument('--create-sample', action='store_true', help='Create and use a tiny sample Keras model if none found')
parser.add_argument('--dry-run', action='store_true', help='Load model and exit (no camera)')
parser.add_argument('--no-sound', action='store_true', help='Disable sound output')
parser.add_argument('--no-overlay', action='store_true', help='Disable on-screen overlays (FPS/labels)')
parser.add_argument('--log-path', help='Path to log file', default=None)
parser.add_argument('--test-alarm', action='store_true', help='Play the alarm once immediately and exit')
parser.add_argument('--debug', action='store_true', help='Enable debug mode (prints and saves crops)')
args = parser.parse_args()


if args.model:
    MODEL_PATH = args.model
if args.camera is not None:
    CAMERA_INDEX = args.camera
if args.alarm:
    ALARM_PATH = args.alarm

if args.debug:
    os.makedirs('debug', exist_ok=True)
    logging.info("Debug mode enabled; saving eye crops to ./debug/")


if not args.no_sound:
    try:
        mixer.init()
        if os.path.exists(ALARM_PATH):
            sound = mixer.Sound(ALARM_PATH)
            logging.info(f"Sound initialized from {ALARM_PATH}")
        else:
            sound = None
            logging.warning(f"Alert sound file not found at {ALARM_PATH}, sound disabled.")
    except Exception as e:
        sound = None
        logging.exception("Warning: mixer init failed, sound disabled:")
else:
    sound = None


if args.log_path:
    LOG_PATH = args.log_path
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f"Logging initialized. Log file: {LOG_PATH}")


if args.test_alarm:
    logging.info("Test alarm requested via --test-alarm")
    played = False
    
    if sound:
        try:
            ch = sound.play()
            logging.info("Called Sound.play() for test alarm")
            played = True
        except Exception as e:
            logging.exception(f"Sound.play() failed during test alarm: {e}")
    
    if not played and os.path.exists(ALARM_PATH):
        try:
            mixer.music.load(ALARM_PATH)
            mixer.music.play()
            logging.info("Played alarm via mixer.music fallback (test)")
            played = True
        except Exception as e:
            logging.exception(f"mixer.music fallback failed during test alarm: {e}")
    
    if not played and os.name == 'nt' and os.path.exists(ALARM_PATH):
        try:
            import winsound
            winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            logging.info("Played alarm via winsound fallback (test)")
            played = True
        except Exception as e:
            logging.exception(f"winsound fallback failed during test alarm: {e}")
    if played:
        print("Test alarm played (check speakers). Exiting.")
        logging.info("Test alarm playback succeeded; exiting.")
        sys.exit(0)
    else:
        print("Test alarm failed: no method could play the alarm. See logs for details.")
        logging.error("Test alarm playback failed; no available playback method")
        sys.exit(1)


candidates = []
if os.path.exists(MODEL_PATH):
    candidates.append(MODEL_PATH)

for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith('.h5'):
            p = os.path.join(root, f)
            if p not in candidates:
                candidates.append(p)

model = None
loaded_model_path = None
for c in candidates:
    try:
        print(f"Trying to load model: {c}")
        model = load_model(c)
        loaded_model_path = c
        print(f"Loaded model: {c}")
        break
    except Exception as e:
        print(f"Warning: failed to load model {c}: {e}")

if model is None:
    
    print("No model found. Creating a generic sample model to allow the application to run...")
    logging.warning("No .h5 model found. Generating a dummy model. ACCURACY WILL BE LOW until a real model is provided.")
    
    def create_and_save_sample(path):
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            model_sample = Sequential([
                Input(shape=(24,24,1)),
                Conv2D(32, (3,3), activation='relu'),
                MaxPooling2D(),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(2, activation='softmax'),
            ])
            model_sample.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model_sample.save(path)
            logging.info(f"Sample model created at {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to create sample model: {e}")
            return False

    if create_and_save_sample(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            loaded_model_path = MODEL_PATH
        except Exception as e:
            logging.error(f"Failed to load the just-created sample model: {e}")

if model is None:
    print(f"CRITICAL ERROR: No valid Keras model found and could not create one. Please place a Keras .h5 model at {MODEL_PATH}.")
    
    pass

model_path = loaded_model_path
logging.info(f"Model loaded from: {model_path}")

if args.dry_run:
    print(f"Dry run complete. Model loaded from {model_path}.")
    sys.exit(0)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}")
    sys.exit(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred = np.array([99])
lpred = np.array([99])
# FPS / display helpers
last_time = time.time()
fps = 0.0
last_label = ''
last_confidence = 0.0
# alert cooldown
last_alert_time = 0.0

def trigger_alert(frame, score, last_alert_time, force=False):
    """Play the alarm (with fallbacks), write snapshot, and return updated alert time."""
    cv2.imwrite(os.path.join(path,'image.jpg'), frame)
    now = time.time()
    if force or now - last_alert_time >= ALERT_COOLDOWN:
        logging.warning(f"ALERT: sleepy score reached: {score}")
        played = False
        
        if sound:
            try:
                ch = sound.play()
                if ch is None:
                    logging.warning("pygame.mixer.Sound.play() returned None — attempting mixer.music fallback")
                    try:
                        mixer.music.load(ALARM_PATH)
                        mixer.music.play()
                        logging.info("Played alarm via mixer.music fallback")
                        played = True
                    except Exception as e2:
                        logging.exception(f"mixer.music fallback failed: {e2}")
                else:
                    played = True
            except Exception as e:
                logging.exception(f"pygame Sound.play failed: {e}")
        
        if not played and os.path.exists(ALARM_PATH) and os.name == 'nt':
            try:
                import winsound
                winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
                logging.info("Played alarm via winsound fallback")
                played = True
            except Exception as e2:
                logging.exception(f"winsound fallback failed: {e2}")
        if not played:
            logging.error("Failed to play alarm via any method")
        return now
    else:
        logging.debug(f"Alert suppressed by cooldown. Score: {score}")
        return last_alert_time

while(True):
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    rpred = None
    lpred = None
    r_conf = None
    l_conf = None
    r_bbox = None
    l_bbox = None
    height,width = frame.shape[:2]

    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        inst_fps = 1.0 / dt
        fps = fps * FPS_SMOOTHING + inst_fps * (1 - FPS_SMOOTHING)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not args.no_overlay:
        cv2.putText(frame, f"FPS:{fps:.1f}", (width-140,30), font, 1,(255,255,0),1,cv2.LINE_AA)
        cv2.putText(frame, f"{last_label} {last_confidence:.2f}", (10,40), font, 1,(0,255,0),1,cv2.LINE_AA) 

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye_gray = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye_resized = cv2.resize(r_eye_gray,(24,24))
        r_eye_proc = r_eye_resized.astype('float32') / 255.0
        r_eye_proc = r_eye_proc.reshape(1,24,24,1)
        if model:
            preds = model.predict(r_eye_proc)
            rpred = int(np.argmax(preds, axis=1)[0])
            r_conf = float(np.max(preds))
        else:
            rpred = 0 
            rpred = 1 
            r_conf = 0.0

        r_bbox = (x,y,w,h)
        label_r = 'Open' if rpred==1 else 'Closed'
        last_label = f"R:{label_r}"
        last_confidence = r_conf
        if args.debug:
            try:
                cv2.imwrite(os.path.join('debug', f"r_{int(time.time()*1000)}_p{rpred}_c{int(r_conf*100)}.jpg"), r_eye)
            except Exception:
                logging.exception("Failed to save debug right eye crop")
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(frame, f"R:{label_r} {r_conf:.2f}", (x,y-5), font, 0.5, (0,255,0),1,cv2.LINE_AA)
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye_gray = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye_resized = cv2.resize(l_eye_gray,(24,24))
        l_eye_proc = l_eye_resized.astype('float32') / 255.0
        l_eye_proc = l_eye_proc.reshape(1,24,24,1)
        if model:
            preds = model.predict(l_eye_proc)
            lpred = int(np.argmax(preds, axis=1)[0])
            l_conf = float(np.max(preds))
        else:
            lpred = 1
            l_conf = 0.0
            
        l_bbox = (x,y,w,h)
        label_l = 'Open' if lpred==1 else 'Closed'
        last_label = f"L:{label_l}"
        last_confidence = l_conf
        if args.debug:
            try:
                cv2.imwrite(os.path.join('debug', f"l_{int(time.time()*1000)}_p{lpred}_c{int(l_conf*100)}.jpg"), l_eye)
            except Exception:
                logging.exception("Failed to save debug left eye crop")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 1)
        cv2.putText(frame, f"L:{label_l} {l_conf:.2f}", (x,y-5), font, 0.5, (255,255,0),1,cv2.LINE_AA)
        break

    if rpred is not None and lpred is not None:
        if (rpred==0 and lpred==0):
            score = score + SCORE_INCREMENT
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            score = max(0, score - SCORE_DECREMENT)
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        
        score = max(0, score - SCORE_DECREMENT)
        cv2.putText(frame,"Detecting...",(10,height-20), font, 1,(200,200,200),1,cv2.LINE_AA)

    if args.debug:
        logging.info(f"Frame debug: rpred={rpred} r_conf={r_conf} lpred={lpred} l_conf={l_conf} score={score}")

    if(score<0):
        score=0
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if score > ALERT_THRESHOLD:
        last_alert_time = trigger_alert(frame, score, last_alert_time, force=False)
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        logging.info("Manual alert triggered by 'a' key")
        last_alert_time = trigger_alert(frame, score, last_alert_time, force=True)
cap.release()
cv2.destroyAllWindows()