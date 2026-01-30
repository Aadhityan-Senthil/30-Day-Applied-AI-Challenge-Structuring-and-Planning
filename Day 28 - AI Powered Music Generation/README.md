# Day 28: AI-Powered Music Generation

## Overview
Generate original music using LSTM neural networks. Creates MIDI files that can be played with any MIDI player.

## Features
- LSTM-based melody generation
- Musical vocabulary (3 octaves, 12 notes each)
- Temperature-controlled creativity
- MIDI file export
- Piano roll visualization

## Requirements
```bash
pip install numpy matplotlib midiutil
pip install tensorflow  # For neural network
```

## Usage
```bash
python music_generation.py
```

## Output Files
- `generated_melody_*.mid` - Playable MIDI files
- `melody_visualization.png` - Piano roll visualization
- `music_generation_results.png` - Training and generation stats

## Temperature Parameter
- **Low (0.5-0.8)**: More predictable, musical melodies
- **Medium (0.8-1.0)**: Balanced creativity
- **High (1.0+)**: More experimental, random

## Play Generated Music
- Windows: Windows Media Player
- macOS: GarageBand, QuickTime
- Linux: TiMidity++, VLC
- Online: https://onlinesequencer.net/import

## Extend with Real Data
Download MIDI datasets:
```
https://www.kaggle.com/datasets
Search for "MIDI music dataset"
```
