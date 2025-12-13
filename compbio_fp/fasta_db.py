"""FASTA database utilities for protein folding demo"""

import os
import glob

def parse_fasta(filepath):
    """Parse a FASTA file and return sequences as dict {header: sequence}"""
    sequences = {}
    current_header = None
    current_seq = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header:
                        sequences[current_header] = ''.join(current_seq)
                    current_header = line[1:]  # Remove '>'
                    current_seq = []
                elif line:
                    current_seq.append(line)
            
            if current_header:
                sequences[current_header] = ''.join(current_seq)
    except Exception:
        pass
    
    return sequences

def load_database_sequences():
    """Load all FASTA sequences from the databases folder"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'databases')
    sequences = {}
    
    # Find all FASTA files
    fasta_files = glob.glob(os.path.join(db_path, '*.fasta')) + glob.glob(os.path.join(db_path, '*.fasta.txt'))
    
    for fasta_file in fasta_files:
        file_sequences = parse_fasta(fasta_file)
        sequences.update(file_sequences)
    
    return sequences