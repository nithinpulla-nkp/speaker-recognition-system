#!/usr/bin/env python3
"""
Speaker Recognition System using MFCC features and GMM/SVM models
Based on the original project by Nithin P and team (2018-2022)
"""

import os
import pickle
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from python_speech_features import mfcc, delta
import warnings
warnings.filterwarnings("ignore")


class SpeakerRecognition:
    """Main class for speaker recognition system"""
    
    def __init__(self, num_components=16, covariance_type='diag'):
        """
        Initialize the speaker recognition system
        
        Args:
            num_components: Number of GMM components
            covariance_type: Type of covariance for GMM
        """
        self.num_components = num_components
        self.covariance_type = covariance_type
        self.gmm_models = {}
        self.svm_model = None
        self.speakers = []
        self.features_scaler = None
        
    def extract_features(self, audio_path, sr=22050):
        """
        Extract MFCC and delta MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Combined MFCC and delta features (40 dimensions)
        """
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_path, sr=sr)
            
            # Extract MFCC features (20 coefficients)
            mfcc_features = mfcc(audio, sample_rate, 
                               numcep=20,      # 20 MFCC coefficients
                               nfilt=26,       # 26 filters
                               nfft=512,       # FFT size
                               lowfreq=0,      # Low frequency
                               highfreq=None,  # High frequency
                               preemph=0.97,   # Pre-emphasis
                               ceplifter=22,   # Cepstral liftering
                               appendEnergy=False)
            
            # Extract delta features (20 coefficients)
            delta_features = delta(mfcc_features, 2)
            
            # Combine MFCC and delta features (40 total features)
            combined_features = np.hstack((mfcc_features, delta_features))
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def prepare_training_data(self, data_dir):
        """
        Prepare training data from directory structure
        Expected structure: data_dir/speaker_name/*.wav
        
        Args:
            data_dir: Directory containing speaker folders
            
        Returns:
            features_dict: Dictionary with speaker names as keys and features as values
        """
        features_dict = {}
        
        for speaker_name in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker_name)
            
            if not os.path.isdir(speaker_path):
                continue
                
            print(f"Processing speaker: {speaker_name}")
            speaker_features = []
            
            # Process all audio files for this speaker
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(speaker_path, audio_file)
                    features = self.extract_features(audio_path)
                    
                    if features is not None:
                        speaker_features.append(features)
            
            if speaker_features:
                # Concatenate all features for this speaker
                features_dict[speaker_name] = np.vstack(speaker_features)
                print(f"Extracted {len(speaker_features)} audio samples for {speaker_name}")
            else:
                print(f"No valid audio files found for {speaker_name}")
        
        self.speakers = list(features_dict.keys())
        return features_dict
    
    def train_gmm_models(self, features_dict):
        """
        Train individual GMM models for each speaker
        
        Args:
            features_dict: Dictionary with speaker features
        """
        print("Training GMM models...")
        
        for speaker, features in features_dict.items():
            print(f"Training GMM for {speaker}...")
            
            # Create and train GMM model
            gmm = GaussianMixture(
                n_components=self.num_components,
                covariance_type=self.covariance_type,
                max_iter=200,
                random_state=42
            )
            
            gmm.fit(features)
            self.gmm_models[speaker] = gmm
            
            print(f"GMM trained for {speaker} with {len(features)} samples")
    
    def train_svm_model(self, features_dict):
        """
        Train SVM model for comparison
        
        Args:
            features_dict: Dictionary with speaker features
        """
        print("Training SVM model...")
        
        # Prepare data for SVM
        X = []
        y = []
        
        for speaker, features in features_dict.items():
            X.extend(features)
            y.extend([speaker] * len(features))
        
        X = np.array(X)
        y = np.array(y)
        
        # Train SVM with RBF kernel
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X, y)
        
        print(f"SVM trained with {len(X)} samples from {len(features_dict)} speakers")
    
    def predict_gmm(self, test_features, threshold=-50):
        """
        Predict speaker using GMM models
        
        Args:
            test_features: Features extracted from test audio
            threshold: Threshold for unknown speaker detection
            
        Returns:
            Predicted speaker name and confidence score
        """
        if not self.gmm_models:
            return "No trained models", 0
        
        scores = {}
        
        # Calculate likelihood scores for each speaker model
        for speaker, gmm in self.gmm_models.items():
            try:
                score = gmm.score(test_features)
                scores[speaker] = score
            except:
                scores[speaker] = -np.inf
        
        # Find the best score
        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]
        
        # Check if score is above threshold (unknown speaker detection)
        if best_score < threshold:
            return "Unknown", best_score
        
        return best_speaker, best_score
    
    def predict_svm(self, test_features):
        """
        Predict speaker using SVM model
        
        Args:
            test_features: Features extracted from test audio
            
        Returns:
            Predicted speaker name and confidence score
        """
        if self.svm_model is None:
            return "No trained model", 0
        
        # Get prediction and probability
        prediction = self.svm_model.predict(test_features.mean(axis=0).reshape(1, -1))
        probabilities = self.svm_model.predict_proba(test_features.mean(axis=0).reshape(1, -1))
        
        best_speaker = prediction[0]
        confidence = np.max(probabilities)
        
        return best_speaker, confidence
    
    def save_models(self, models_dir="models"):
        """Save trained models to disk"""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save GMM models
        for speaker, gmm in self.gmm_models.items():
            model_path = os.path.join(models_dir, f"{speaker}.gmm")
            with open(model_path, 'wb') as f:
                pickle.dump(gmm, f)
        
        # Save SVM model
        if self.svm_model:
            svm_path = os.path.join(models_dir, "svm_model.pkl")
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_model, f)
        
        # Save speaker list
        speakers_path = os.path.join(models_dir, "speakers.pkl")
        with open(speakers_path, 'wb') as f:
            pickle.dump(self.speakers, f)
        
        print(f"Models saved to {models_dir}/")
    
    def load_models(self, models_dir="models"):
        """Load trained models from disk"""
        try:
            # Load speaker list
            speakers_path = os.path.join(models_dir, "speakers.pkl")
            with open(speakers_path, 'rb') as f:
                self.speakers = pickle.load(f)
            
            # Load GMM models
            for speaker in self.speakers:
                model_path = os.path.join(models_dir, f"{speaker}.gmm")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.gmm_models[speaker] = pickle.load(f)
            
            # Load SVM model
            svm_path = os.path.join(models_dir, "svm_model.pkl")
            if os.path.exists(svm_path):
                with open(svm_path, 'rb') as f:
                    self.svm_model = pickle.load(f)
            
            print(f"Models loaded from {models_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


def main():
    """Main function to demonstrate the speaker recognition system"""
    
    # Initialize the system
    sr_system = SpeakerRecognition()
    
    print("Speaker Recognition System")
    print("=" * 50)
    
    choice = input("1. Train new models\n2. Test existing models\nChoose option (1/2): ")
    
    if choice == "1":
        # Training mode
        data_dir = input("Enter path to training data directory: ")
        
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist!")
            return
        
        # Prepare training data
        features_dict = sr_system.prepare_training_data(data_dir)
        
        if not features_dict:
            print("No training data found!")
            return
        
        # Train models
        sr_system.train_gmm_models(features_dict)
        sr_system.train_svm_model(features_dict)
        
        # Save models
        sr_system.save_models()
        print("Training completed and models saved!")
        
    elif choice == "2":
        # Testing mode
        if not sr_system.load_models():
            print("Could not load models. Please train first.")
            return
        
        test_audio = input("Enter path to test audio file: ")
        
        if not os.path.exists(test_audio):
            print(f"Audio file {test_audio} does not exist!")
            return
        
        # Extract features from test audio
        test_features = sr_system.extract_features(test_audio)
        
        if test_features is None:
            print("Could not extract features from test audio!")
            return
        
        # Make predictions
        gmm_speaker, gmm_score = sr_system.predict_gmm(test_features)
        svm_speaker, svm_score = sr_system.predict_svm(test_features)
        
        print("\nPrediction Results:")
        print(f"GMM Model: {gmm_speaker} (Score: {gmm_score:.2f})")
        print(f"SVM Model: {svm_speaker} (Confidence: {svm_score:.2f})")
        
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()