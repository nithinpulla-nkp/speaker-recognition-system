#!/usr/bin/env python3
"""
Test script for speaker recognition system
"""

import os
import numpy as np
from speaker_recognition import SpeakerRecognition


def batch_test(sr_system, test_dir, output_file="test_results.txt"):
    """
    Test the system on a batch of test files
    
    Args:
        sr_system: Trained SpeakerRecognition instance
        test_dir: Directory containing test audio files
        output_file: File to save results
    """
    results = []
    
    print("Running batch test...")
    
    for audio_file in os.listdir(test_dir):
        if audio_file.endswith(('.wav', '.mp3', '.flac')):
            audio_path = os.path.join(test_dir, audio_file)
            
            # Extract features
            test_features = sr_system.extract_features(audio_path)
            
            if test_features is None:
                continue
            
            # Make predictions
            gmm_speaker, gmm_score = sr_system.predict_gmm(test_features)
            svm_speaker, svm_score = sr_system.predict_svm(test_features)
            
            result = {
                'file': audio_file,
                'gmm_prediction': gmm_speaker,
                'gmm_score': gmm_score,
                'svm_prediction': svm_speaker,
                'svm_confidence': svm_score
            }
            
            results.append(result)
            print(f"{audio_file}: GMM={gmm_speaker} ({gmm_score:.2f}), SVM={svm_speaker} ({svm_score:.2f})")
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write("File\tGMM_Prediction\tGMM_Score\tSVM_Prediction\tSVM_Confidence\n")
        for result in results:
            f.write(f"{result['file']}\t{result['gmm_prediction']}\t{result['gmm_score']:.4f}\t"
                   f"{result['svm_prediction']}\t{result['svm_confidence']:.4f}\n")
    
    print(f"Results saved to {output_file}")
    return results


def calculate_accuracy(results, ground_truth):
    """
    Calculate accuracy given ground truth labels
    
    Args:
        results: List of prediction results
        ground_truth: Dictionary mapping filenames to true speaker names
    """
    gmm_correct = 0
    svm_correct = 0
    total = 0
    
    for result in results:
        filename = result['file']
        if filename in ground_truth:
            true_speaker = ground_truth[filename]
            
            if result['gmm_prediction'] == true_speaker:
                gmm_correct += 1
            if result['svm_prediction'] == true_speaker:
                svm_correct += 1
            
            total += 1
    
    if total > 0:
        gmm_accuracy = (gmm_correct / total) * 100
        svm_accuracy = (svm_correct / total) * 100
        
        print(f"\nAccuracy Results:")
        print(f"GMM Accuracy: {gmm_accuracy:.2f}% ({gmm_correct}/{total})")
        print(f"SVM Accuracy: {svm_accuracy:.2f}% ({svm_correct}/{total})")
        
        return gmm_accuracy, svm_accuracy
    
    return 0, 0


def main():
    # Initialize system
    sr_system = SpeakerRecognition()
    
    # Load trained models
    if not sr_system.load_models():
        print("Could not load models. Please train first using speaker_recognition.py")
        return
    
    print("Available speakers:", sr_system.speakers)
    
    choice = input("\n1. Single file test\n2. Batch test\n3. Accuracy evaluation\nChoose option: ")
    
    if choice == "1":
        # Single file test
        test_file = input("Enter path to test audio file: ")
        
        if not os.path.exists(test_file):
            print("File does not exist!")
            return
        
        test_features = sr_system.extract_features(test_file)
        if test_features is None:
            print("Could not extract features!")
            return
        
        gmm_speaker, gmm_score = sr_system.predict_gmm(test_features)
        svm_speaker, svm_score = sr_system.predict_svm(test_features)
        
        print(f"\nResults for {os.path.basename(test_file)}:")
        print(f"GMM: {gmm_speaker} (score: {gmm_score:.4f})")
        print(f"SVM: {svm_speaker} (confidence: {svm_score:.4f})")
        
    elif choice == "2":
        # Batch test
        test_dir = input("Enter path to test directory: ")
        
        if not os.path.exists(test_dir):
            print("Directory does not exist!")
            return
        
        results = batch_test(sr_system, test_dir)
        print(f"Tested {len(results)} files")
        
    elif choice == "3":
        # Accuracy evaluation
        test_dir = input("Enter path to test directory: ")
        ground_truth_file = input("Enter path to ground truth file (filename\\tspeaker format): ")
        
        if not os.path.exists(test_dir) or not os.path.exists(ground_truth_file):
            print("Directory or ground truth file does not exist!")
            return
        
        # Load ground truth
        ground_truth = {}
        with open(ground_truth_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ground_truth[parts[0]] = parts[1]
        
        # Run tests
        results = batch_test(sr_system, test_dir)
        
        # Calculate accuracy
        calculate_accuracy(results, ground_truth)
        
    else:
        print("Invalid option!")


if __name__ == "__main__":
    main()