#### Bbliotheken ####
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
import cv2
import adresses.adresses as ad
import matplotlib.pyplot as plt
from collections import deque
import time
import random
import os
import tqdm
from datetime import datetime
import threading

# Deep-Q Network Bibliotheken
import tensorflow as tf
from keras.api.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Activation
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.callbacks import TensorBoard

# TensorFlow GPU- und CPU-Konfiguration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

#### Gameboy Emulator ####
from pyboy import PyBoy
rom_path = "..\Pokemon_Silver_Game\Pokemon_Silver.gbc"
emulation_speed = int(input("Was für eine Geschwindigkeit willst du haben?(0=Unendlich, 1=1fach, 2=2fach.....): "))

#### Globale Variablen ####
action_map = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
game_window_size = (144, 160, 1)

replay_memory_size = 50_000 # 50000, Unterstrich zur besseren Lesbarkeit
min_replay_memory_size = 1000
model_name = "128x1"
minibatch_size = 32
discount = 0.99
update_target_every = 5
episodes = 1
epsilon = 1
epsilon_decay = 0.99975
min_epsilon = 0.001
aggregate_stats_every = 50 # episodes
min_reward = 0

#### Hauptprogramm ####
## Umgebungsdefinition ##
class PokemonEnv (gym.Env):
    
    def __init__(self):
        super().__init__()  # Initialisiere die Oberklasse gym.Env
        self.pyboy = PyBoy(rom_path, window="SDL2") # Initialisieren des Emulators
        self.pyboy.set_emulation_speed(emulation_speed) # Festlegen der Emulations-Geschwindigkeit
        # Festlegen des Aktionsraumes
        self.action_space = spaces.Discrete(len(action_map))
        # Low = Niedrigster Wert, High = Höchster Wert, shape = Form des Beobachtungsraumes + Farbraum, dtype = Datentyp
        self.observation_space = Box(low=0, high= 255, shape=game_window_size, dtype=np.uint8)

        # Variablen der Umgebung festlegen
        self.counter = 0
        self.done = 0    
        self.reward = 0 
        self.total_reward = 0
        self.obs = 0
        self.positions = []
    
    def step(self, action):
        # Umwandeln der ausgegebenen Nummer in eine entsprechende Aktion
        if action == 0:
            pass
        else:
            self.pyboy.button(action_map[action])



    def close(self):
        cv2.destroyAllWindows()

    def get_done(self):
        # Definieren der Done Variable
        done = False
        # 1 = Kampf verloren, Spielende wird eingeleitet, 2 = Weggelaufen
        battle_result = self.pyboy.memory[ad.battle_result]
        if battle_result == 1:
            done = True

        return done
    
    def get_reward(self):
        reward = 0
        position_x = self.pyboy.memory[ad.player_x]
        position_y = self.pyboy.memory[ad.player_y]

        if (position_x, position_y) in self.positions:
            pass
        elif (position_x, position_y) not in self.positions:
            self.positions.append((position_x, position_y))
            reward += 1
        else:
            print("Was programmierst du da denn schon wieder????!!!!!")

        return reward
    # Funktion um das Bild des Emulators einzufangen
    def get_observation(self):
        return self.convert_to_grayscale(self.pyboy.screen.ndarray)
    
    # Funktion um das Observations-Bild auszugeben
    def show(self):
        # Anzeigen des Bildes der Funktion "get_observation" in einem seperaten Fenster
        plt.imshow(self.get_observation(), cmap="gray") # Colormap gray wird verwendet um zu visualisieren was die KI sieht
        plt.colorbar()  # Hinzufügen einer Farblegende basierend auf der Colormap
        plt.title('Game Area Visualization') # Fenstername festelgen
        plt.show() # Anzeigen des Fenster mit den eingestellten Einstellungen

    def convert_to_grayscale(self, image):
        # Überprüfen ob das Eingabebilg einen vierten Kanal besitzt
        if image.shape[2] == 4:
            image = image[:, :, :3] # Abschneiden des vierten Kanals
        # Ausgabe Variable auf das umgewandelte 3-Kanal Bild setzen
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return gray_image # Rückgbabe des neuen Graustufenbildes
    
    def log(self):
        self.counter += 1
        # Emulationsgeschwindigkeit Maximum = gib jeden Frame den Wert aus
        if emulation_speed == 0:
            print(f"Total Reward: {self.total_reward}")
        # Emulationsgeschwindigkeit > 0 = gib alle 60 Frames ~ jede Sekunde den Wert aus
        elif self.counter == 60 * emulation_speed:
            print(f"Total Reward: {self.total_reward}")
            self.counter = 0
        
        pass

## Definition des künstlichen Intelligenz-Agenten ##
class DQNAgent():
    def __init__(self):
        # Main Model # gets trained every step
        self.model = self.create_model()

        # Target Model # this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)
        current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{model_name}-{current_date}")

        # Zähler wann das Programm bereit ist das Target Model zu aktualisieren 
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(128, (3,3), input_shape=game_window_size))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(32))

        model.add(Dense(PokemonEnv().action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state))
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < min_replay_memory_size:
            return
        
        minibatch = random.sample(self.replay_memory, minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size = minibatch_size, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

## Modifiziertes Tensorboard ##
# Inspirations-Quelle: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ 14.07.2024, 12:51 Uhr
class ModifiedTensorBoard(TensorBoard):

    def __init__(self,log_dir ="./logs", **kwargs):
        # Aufruf des Konstruktors der Basisklasse # Jeden Batch wird aktualisiert
        super().__init__(log_dir=log_dir, update_freq="batch", **kwargs)
        # Zähler, wie viele Batches verarbeitet wurden
        self.step = 1
        # Erstellen eines Dateischreibers um TensorBoard Log Files im angegebenen Verzeichnis zu erstellen
        self.writer = tf.summary.create_file_writer(self.log_dir)

    #Setzt das Modell für den Callback, aber verhindert die automatische Erstellung von Log-Schreibern.
    def set_model(self, model):
        # Modell für den Callback speichern
        self.model = model


    def on_epoch_end(self, epoch, logs=None):
        """
        Würde am Ende jeder Epoche aufgerufen werden, da die Logging-Funktion für jede Batch durchgeführt wird, passiert dies nicht.
        """

    def on_batch_end(self, batch, logs=None):
        """
        Protokolliert nach jeder Batch die Trainingsmetriken
        Args:
            batch: Die Nummer der aktuellen Batch innerhalb der aktuellen Epoche.
            logs: Ein Dictionary, das die Trainingsmetriken enthält.
        """
        # Log-Daten schreiben
        self._write_logs(logs, self.step)
        # Erhöhen der Schrittvariable nach jeder Batch
        self.step += 1

    # Overrided, so won't close writer
    def on_train_end(self):
        """
        Wird aufgerufen, wenn das Training beendet ist. Schließt den TensorBoard-Writer.
        """
        self.writer.close()

    def _write_logs(self, logs, step):
        """
        Schreibt die angegebenen Logs in die TensorBoard-Daten.
        Args:
            logs: Ein Dictionary von Metriken, die protokolliert werden sollen.
            step: Die aktuelle Schrittnummer im Trainingsprozess.
        """
        with self.writer.as_default():
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                tf.summary.scalar(name, value, step=step)
            self.writer.flush()

#### Hauptschleife ####
# Initialisierung der Umgebung und des Agenten.
agent = DQNAgent()

# Festlegen eines festen Zufallschemas für die Reproduzierbarkeit der Ergebnisse.
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Initialisierung einer Liste, um Belohnungen von Episoden zu speichern.
ep_rewards = [-200]

# Überprüfen, ob ein Verzeichnis für die gespeicherten Modelle existiert. Wenn nicht, wird es erstellt.
if not os.path.isdir("models"):
    os.makedirs("models")

#### Hauptschleife des Trainingsprozesses ####
for episode in tqdm.tqdm(range(1, episodes+1), ascii=True, unit="episodes"):

    # Fortsetzung der Episode, bis das Ende-Signal (env.done) von der Umgebung gesetzt wird.
    done = False
    while not done:
        env = PokemonEnv()
        # Setzt die aktuelle Episode im TensorBoard für das Logging.
        agent.tensorboard.step = episode
        episode_reward = 0  # Summe der Belohnungen in der aktuellen Episode.
        step = 1  # Schrittzähler innerhalb der aktuellen Episode.
        current_state = env.get_observation()  # Ersten Zustand der Umgebung abrufen.
        while env.pyboy.tick():
            # Entscheidung, ob eine zufällige Aktion gewählt wird (Exploration) oder die beste bekannte Aktion (Exploitation).
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Ausführen der gewählten Aktion in der Umgebung.
            env.step(action)
            new_state = env.get_observation()  # Neuen Zustand nach der Aktion abrufen.
            env.reward = env.get_reward()  # Belohnung für die ausgeführte Aktion erhalten.
            done = env.get_done()  # Überprüfen, ob die Episode beendet ist.

            # Zusammenrechnen der Belohnungen der aktuellen Episode.
            episode_reward += env.reward

            # Aktualisieren des Erfahrungsspeichers (Replay Memory).
            agent.update_replay_memory((current_state, action, env.reward, new_state, env.done))
            # Training des Modells, falls nötig.
            agent.train(env.done, step)

            # Aktualisieren des Zustands für den nächsten Schritt.
            current_state = new_state
            step += 1

            if done == True:
                # Anhalten des Emulators am Ende der Episode.
                env.pyboy.stop()

    # Speichern der gesammelten Episode-Belohnung.
    ep_rewards.append(episode_reward)
    # Berechnung und Logging der Statistiken nach jeder Episode oder gemäß des festgelegten Intervalls.
    if not episode % aggregate_stats_every or episode == 1:
        average_reward = sum(ep_rewards[-aggregate_stats_every:]) / len(ep_rewards[-aggregate_stats_every:])
        min_reward = min(ep_rewards[-aggregate_stats_every:])
        max_reward = max(ep_rewards[-aggregate_stats_every:])
        agent.tensorboard._write_logs(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Modell speichern, wenn die durchschnittliche Belohnung einen bestimmten Schwellenwert überschreitet.
        if average_reward >= min_reward:
            agent.model.save(f'models/{model_name}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        