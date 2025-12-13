"""Multi-scale optimization for improved protein folding accuracy"""

import numpy as np
from .optimizer import SimulatedAnnealer
from .energy import EnergyFunction

class MultiScaleOptimizer:
    """Multi-scale optimization using coarse-to-fine approach for better accuracy"""
    
    def __init__(self, protein, energy_fn_class=EnergyFunction, debug=False):
        self.protein = protein
        self.energy_fn_class = energy_fn_class
        self.debug = debug
        
    def run(self, max_steps=25000):
        """Fast 3-stage multi-scale optimization"""
        
        # Stage 1: Coarse exploration
        coarse_energy = self.energy_fn_class(self.protein)
        original_total = coarse_energy.total_energy
        def coarse_total_energy():
            result = original_total()
            result['bond'] *= 0.3
            result['angle'] *= 0.3
            result['hydro'] *= 3.0
            result['compact'] *= 3.0
            result['ss'] *= 0.8
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        coarse_energy.total_energy = coarse_total_energy
        
        coarse_optimizer = SimulatedAnnealer(
            self.protein, coarse_energy,
            temp_K=3000.0, cooling=0.990, max_steps=max_steps//3,
            step_size=0.15, adaptive_cooling=True
        )
        
        coords1, energy1, history1 = coarse_optimizer.run()
        self.protein.coords = coords1
        
        # Stage 2: Medium refinement
        medium_energy = self.energy_fn_class(self.protein)
        def medium_total_energy():
            result = original_total()
            result['hydro'] *= 2.0
            result['compact'] *= 2.0
            result['ss'] *= 1.5
            if 'hbond' in result: result['hbond'] *= 1.5
            if 'torsion' in result: result['torsion'] *= 1.2
            if 'loop' in result: result['loop'] *= 1.2
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        medium_energy.total_energy = medium_total_energy
        
        medium_optimizer = SimulatedAnnealer(
            self.protein, medium_energy,
            temp_K=1500.0, cooling=0.994, max_steps=max_steps//3,
            step_size=0.10, adaptive_cooling=True
        )
        
        coords2, energy2, history2 = medium_optimizer.run()
        self.protein.coords = coords2
        
        # Stage 3: Fine optimization
        fine_energy = self.energy_fn_class(self.protein)
        
        fine_optimizer = SimulatedAnnealer(
            self.protein, fine_energy,
            temp_K=800.0, cooling=0.996, max_steps=max_steps//3,
            step_size=0.06, adaptive_cooling=True
        )
        
        coords3, energy3, history3 = fine_optimizer.run()
        
        combined_history = history1 + history2 + history3
        return coords3, energy3, combined_history

class AdaptiveOptimizer:
    """Adaptive optimizer that adjusts strategy based on sequence properties"""
    
    def __init__(self, protein, energy_fn_class=EnergyFunction, debug=False):
        self.protein = protein
        self.energy_fn_class = energy_fn_class
        self.debug = debug
        
    def analyze_sequence(self):
        """Analyze sequence to determine optimal folding strategy"""
        sequence = self.protein.sequence
        
        # Calculate hydrophobic content
        hydrophobic_residues = set('AILMFWYV')
        hydrophobic_fraction = sum(1 for aa in sequence if aa in hydrophobic_residues) / len(sequence)
        
        # Calculate charged content
        charged_residues = set('DEKR')
        charged_fraction = sum(1 for aa in sequence if aa in charged_residues) / len(sequence)
        
        # Determine strategy
        if hydrophobic_fraction > 0.4:
            strategy = 'hydrophobic_collapse'
        elif charged_fraction > 0.3:
            strategy = 'electrostatic_guided'
        else:
            strategy = 'balanced'
            
        return {
            'strategy': strategy,
            'hydrophobic_fraction': hydrophobic_fraction,
            'charged_fraction': charged_fraction,
            'length': len(sequence)
        }
    
    def run(self, max_steps=25000):
        """Run adaptive optimization based on sequence analysis"""
        
        analysis = self.analyze_sequence()
        strategy = analysis['strategy']
        
        if self.debug:
            print(f"Using strategy: {strategy}")
            print(f"Hydrophobic fraction: {analysis['hydrophobic_fraction']:.2f}")
            print(f"Charged fraction: {analysis['charged_fraction']:.2f}")
        
        if strategy == 'hydrophobic_collapse':
            return self._hydrophobic_strategy(max_steps)
        elif strategy == 'electrostatic_guided':
            return self._electrostatic_strategy(max_steps)
        else:
            return self._balanced_strategy(max_steps)
    
    def _hydrophobic_strategy(self, max_steps):
        """Ultra-enhanced hydrophobic strategy for maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Ultra-enhanced energy function for hydrophobic proteins
        original_total = energy_fn.total_energy
        def hydrophobic_total_energy():
            result = original_total()
            result['hydro'] *= 4.5  # Much stronger hydrophobic driving force
            result['compact'] *= 3.5  # Much stronger compactness
            result['lj'] *= 2.2  # Enhanced LJ interactions
            result['ss'] *= 2.0  # Much stronger secondary structure
            if 'hbond' in result: result['hbond'] *= 1.8  # Enhanced hydrogen bonding
            if 'rama' in result: result['rama'] *= 2.5  # Strong Ramachandran bias
            if 'torsion' in result: result['torsion'] *= 1.5  # Enhanced torsional constraints
            if 'loop' in result: result['loop'] *= 1.3  # Better loop closure
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = hydrophobic_total_energy
        
        # Fast 2-stage optimization
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2500.0, cooling=0.994, max_steps=max_steps//2,
            step_size=0.10, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1000.0, cooling=0.997, max_steps=max_steps//2,
            step_size=0.06, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history
    
    def _electrostatic_strategy(self, max_steps):
        """Ultra-enhanced strategy for charged proteins with maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Ultra-enhanced energy function for charged proteins
        original_total = energy_fn.total_energy
        def electrostatic_total_energy():
            result = original_total()
            result['elec'] *= 3.5  # Much stronger electrostatics
            if 'hbond' in result: result['hbond'] *= 2.5  # Enhanced hydrogen bonding for charged residues
            result['compact'] *= 0.6  # Reduced compactness to allow charge separation
            result['ss'] *= 1.8  # Enhanced secondary structure
            if 'rama' in result: result['rama'] *= 2.0  # Strong backbone constraints
            if 'torsion' in result: result['torsion'] *= 1.3  # Enhanced torsional bias
            if 'loop' in result: result['loop'] *= 1.5  # Better loop formation for charge accommodation
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = electrostatic_total_energy
        
        # Multi-stage optimization for charged proteins
        # Stage 1: Extended conformation search
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2800.0, cooling=0.990, max_steps=max_steps//2,
            step_size=0.14, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        # Stage 2: Fine electrostatic optimization
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1000.0, cooling=0.995, max_steps=max_steps//2,
            step_size=0.08, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history
    
    def _balanced_strategy(self, max_steps):
        """Ultra-enhanced balanced strategy for mixed sequences with maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Enhanced balanced energy function
        original_total = energy_fn.total_energy
        def balanced_total_energy():
            result = original_total()
            # Balanced enhancement of all terms for maximum accuracy
            result['hydro'] *= 2.8
            result['elec'] *= 2.2
            result['compact'] *= 2.0
            result['ss'] *= 2.5
            if 'hbond' in result: result['hbond'] *= 2.0
            if 'rama' in result: result['rama'] *= 2.8
            if 'torsion' in result: result['torsion'] *= 1.8
            if 'loop' in result: result['loop'] *= 1.6
            result['lj'] *= 1.5
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = balanced_total_energy
        
        # Fast 2-stage balanced optimization
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2800.0, cooling=0.992, max_steps=max_steps//2,
            step_size=0.12, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1200.0, cooling=0.996, max_steps=max_steps//2,
            step_size=0.06, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history

__all__ = ['MultiScaleOptimizer', 'AdaptiveOptimizer']