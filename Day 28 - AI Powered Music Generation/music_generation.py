"""
Day 28: AI-Powered Music Generation
30-Day AI Challenge

Generate music using LSTM neural networks.
Creates MIDI files that can be played with any MIDI player.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from midiutil import MIDIFile
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Musical constants
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = [3, 4, 5]
DURATIONS = [0.25, 0.5, 1.0, 2.0]  # Quarter, half, whole, etc.

def create_note_vocabulary():
    """Create vocabulary of all possible notes."""
    vocab = []
    for octave in OCTAVES:
        for note in NOTES:
            vocab.append(f"{note}{octave}")
    vocab.append('REST')  # Add rest
    return vocab

def generate_training_melodies(n_melodies=100, melody_length=32):
    """Generate synthetic training melodies with musical patterns."""
    np.random.seed(42)
    vocab = create_note_vocabulary()
    
    melodies = []
    
    # Common chord progressions (in terms of scale degrees)
    progressions = [
        [0, 4, 5, 3],  # I-V-vi-IV
        [0, 5, 3, 4],  # I-vi-IV-V
        [0, 3, 4, 4],  # I-IV-V-V
    ]
    
    # C major scale degrees to notes
    c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    
    for _ in range(n_melodies):
        melody = []
        progression = progressions[np.random.randint(0, len(progressions))]
        
        for chord_idx in progression:
            # Generate notes around the chord
            root = c_major[chord_idx]
            octave = np.random.choice([4, 5])
            
            # Add 8 notes per chord (to make melody_length=32)
            for i in range(melody_length // 4):
                if np.random.random() < 0.1:  # 10% chance of rest
                    melody.append('REST')
                else:
                    # Choose note from chord or passing tone
                    if np.random.random() < 0.7:
                        note = root
                    else:
                        note = np.random.choice(c_major)
                    
                    octave_choice = octave + np.random.choice([-1, 0, 0, 0, 1])
                    octave_choice = max(3, min(5, octave_choice))
                    melody.append(f"{note}{octave_choice}")
        
        melodies.append(melody)
    
    return melodies, vocab

def prepare_sequences(melodies, vocab, seq_length=8):
    """Prepare input sequences for LSTM."""
    note_to_idx = {note: i for i, note in enumerate(vocab)}
    
    X = []
    y = []
    
    for melody in melodies:
        for i in range(len(melody) - seq_length):
            seq_in = [note_to_idx.get(n, 0) for n in melody[i:i+seq_length]]
            seq_out = note_to_idx.get(melody[i+seq_length], 0)
            X.append(seq_in)
            y.append(seq_out)
    
    X = np.array(X)
    y = to_categorical(y, num_classes=len(vocab)) if TF_AVAILABLE else np.array(y)
    
    return X, y, note_to_idx

def build_music_model(vocab_size, seq_length):
    """Build LSTM model for music generation."""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        Embedding(vocab_size, 64, input_length=seq_length),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_melody(model, seed_sequence, note_to_idx, idx_to_note, length=32, temperature=1.0):
    """Generate a new melody using the trained model."""
    vocab_size = len(note_to_idx)
    
    if model is None:
        # Random generation fallback
        return [idx_to_note[np.random.randint(0, vocab_size)] for _ in range(length)]
    
    generated = list(seed_sequence)
    current_seq = [note_to_idx.get(n, 0) for n in seed_sequence]
    
    for _ in range(length):
        # Predict next note
        x = np.array([current_seq])
        pred = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        pred = np.log(pred + 1e-10) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))
        
        # Sample from distribution
        next_idx = np.random.choice(len(pred), p=pred)
        next_note = idx_to_note[next_idx]
        
        generated.append(next_note)
        current_seq = current_seq[1:] + [next_idx]
    
    return generated

def note_to_midi_number(note_str):
    """Convert note string to MIDI number."""
    if note_str == 'REST':
        return None
    
    note = note_str[:-1]
    octave = int(note_str[-1])
    
    note_offsets = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                   'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    return 12 * (octave + 1) + note_offsets.get(note, 0)

def create_midi_file(melody, filename='generated_music.mid', tempo=120):
    """Create MIDI file from melody."""
    if not MIDI_AVAILABLE:
        print("midiutil not installed. Cannot create MIDI file.")
        return False
    
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    volume = 100
    
    midi.addTempo(track, 0, tempo)
    
    for note_str in melody:
        midi_num = note_to_midi_number(note_str)
        duration = np.random.choice(DURATIONS[:2])  # Quarter or half notes
        
        if midi_num is not None:
            midi.addNote(track, channel, midi_num, time, duration, volume)
        
        time += duration
    
    with open(filename, 'wb') as f:
        midi.writeFile(f)
    
    print(f"MIDI file saved: {filename}")
    return True

def visualize_melody(melody, title="Generated Melody"):
    """Visualize melody as a piano roll."""
    vocab = create_note_vocabulary()
    note_to_y = {note: i for i, note in enumerate(vocab)}
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    for i, note in enumerate(melody):
        if note != 'REST' and note in note_to_y:
            y = note_to_y[note]
            ax.barh(y, 1, left=i, height=0.8, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Note')
    ax.set_title(title)
    
    # Set y-axis labels (show every 4th note)
    y_ticks = range(0, len(vocab), 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([vocab[i] for i in y_ticks])
    
    ax.set_xlim(0, len(melody))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('melody_visualization.png', dpi=150)
    plt.close()
    print("Melody visualization saved to 'melody_visualization.png'")

def plot_training_and_samples(history, sample_melodies, vocab):
    """Plot training results and sample melodies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    if history:
        axes[0, 0].plot(history.history['loss'], label='Loss')
        axes[0, 0].plot(history.history['accuracy'], label='Accuracy')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Training history\nnot available', ha='center', va='center')
    
    # Note distribution in generated melody
    if sample_melodies:
        all_notes = [n for m in sample_melodies for n in m if n != 'REST']
        unique, counts = np.unique(all_notes, return_counts=True)
        
        top_n = 15
        sorted_idx = np.argsort(counts)[-top_n:]
        
        axes[0, 1].barh([unique[i] for i in sorted_idx], [counts[i] for i in sorted_idx], color='coral')
        axes[0, 1].set_title('Most Common Notes (Generated)')
        axes[0, 1].set_xlabel('Count')
    
    # Piano roll of first generated melody
    if sample_melodies:
        note_to_y = {note: i for i, note in enumerate(vocab)}
        melody = sample_melodies[0][:32]
        
        for i, note in enumerate(melody):
            if note != 'REST' and note in note_to_y:
                y = note_to_y[note]
                axes[1, 0].barh(y, 1, left=i, height=0.8, color='steelblue')
        
        axes[1, 0].set_title('Generated Melody Piano Roll')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Note Index')
    
    # Melody statistics
    stats_text = """
    Music Generation Statistics:
    
    • Vocabulary Size: {} notes
    • Sequence Length: 8 notes
    • Generated Melodies: {}
    • Average Melody Length: 32 notes
    • Tempo: 120 BPM
    
    Notes include:
    - 3 octaves (3-5)
    - 12 semitones per octave
    - Rest notes for rhythm
    """.format(len(vocab), len(sample_melodies) if sample_melodies else 0)
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Generation Statistics')
    
    plt.tight_layout()
    plt.savefig('music_generation_results.png', dpi=150)
    plt.close()
    print("Results saved to 'music_generation_results.png'")

def main():
    print("=" * 50)
    print("Day 28: AI-Powered Music Generation")
    print("=" * 50)
    
    SEQ_LENGTH = 8
    
    # Create vocabulary
    print("\n[1] Creating note vocabulary...")
    vocab = create_note_vocabulary()
    print(f"Vocabulary size: {len(vocab)} notes")
    
    # Generate training data
    print("\n[2] Generating training melodies...")
    melodies, vocab = generate_training_melodies(n_melodies=200, melody_length=32)
    print(f"Generated {len(melodies)} training melodies")
    
    # Prepare sequences
    print("\n[3] Preparing training sequences...")
    X, y, note_to_idx = prepare_sequences(melodies, vocab, SEQ_LENGTH)
    idx_to_note = {i: note for note, i in note_to_idx.items()}
    print(f"Training sequences: {len(X)}")
    
    # Build model
    print("\n[4] Building LSTM model...")
    model = build_music_model(len(vocab), SEQ_LENGTH)
    if model:
        model.summary()
    
    # Train model
    print("\n[5] Training model...")
    if model:
        history = model.fit(X, y, epochs=30, batch_size=64, validation_split=0.1, verbose=1)
    else:
        history = None
        print("TensorFlow not available. Skipping training.")
    
    # Generate new melodies
    print("\n[6] Generating new melodies...")
    generated_melodies = []
    
    for i in range(5):
        # Use a random seed from training data
        seed_idx = np.random.randint(0, len(melodies))
        seed = melodies[seed_idx][:SEQ_LENGTH]
        
        # Generate with different temperatures
        temp = 0.8 + i * 0.1
        melody = generate_melody(model, seed, note_to_idx, idx_to_note, length=32, temperature=temp)
        generated_melodies.append(melody)
        print(f"  Melody {i+1} (temp={temp:.1f}): {' '.join(melody[:8])}...")
    
    # Create MIDI files
    print("\n[7] Creating MIDI files...")
    for i, melody in enumerate(generated_melodies[:3]):
        filename = f'generated_melody_{i+1}.mid'
        create_midi_file(melody, filename)
    
    # Visualize
    print("\n[8] Creating visualizations...")
    visualize_melody(generated_melodies[0], "Generated Melody #1")
    plot_training_and_samples(history, generated_melodies, vocab)
    
    # Save results
    output = {
        'vocab_size': len(vocab),
        'training_melodies': len(melodies),
        'generated_melodies': len(generated_melodies),
        'sequence_length': SEQ_LENGTH,
        'sample_generated': generated_melodies[0][:16]
    }
    
    with open('music_generation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to 'music_generation_results.json'")
    
    print("\n" + "=" * 50)
    print("Day 28 Complete!")
    print("=" * 50)
    print("\n🎵 Play the .mid files with any MIDI player!")

if __name__ == "__main__":
    main()
