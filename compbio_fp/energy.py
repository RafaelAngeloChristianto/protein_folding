from .models import distance
import numpy as np


class EnergyFunction:
    def __init__(self, protein):
        self.p = protein
        # Initialize Ramachandran probability distributions
        self._init_ramachandran_probs()

    def bond_energy(self, k_bond=50.0, r0=3.8):
        if r0 is None:
            r0 = self.p.ca_distance
        e = 0.0
        for i in range(self.p.N - 1):
            r = distance(self.p.coords[i], self.p.coords[i+1])
            e += 0.5 * k_bond * (r - r0)**2
        return e

    def angle_energy(self, k_angle=4.0, theta0=None):
        """Calculate bond angle energy with reduced force constant.
        
        Further reduced from 8.0 to 4.0 to prevent angle terms from dominating.
        This allows more flexibility during folding and better balance with non-bonded terms.
        """
        import numpy as _np
        from .models import angle_between
        if theta0 is None:
            theta0 = _np.deg2rad(110.0)
        e = 0.0
        for i in range(1, self.p.N - 1):
            # deterministically compute angle; angle_between returns 0.0
            # for degenerate cases (zero-length vectors)
            th = angle_between(self.p.coords[i-1], self.p.coords[i], self.p.coords[i+1])
            diff = th - theta0
            e += 0.5 * k_angle * (diff**2)
        return e

    def lj_nonbonded(self, epsilon=0.1, sigma=3.8, cutoff=10.0):
        e = 0.0
        # include CA-CA pairs
        for i in range(self.p.N):
            for j in range(i+2, self.p.N):
                r = distance(self.p.coords[i], self.p.coords[j])
                if r > cutoff or r == 0:
                    continue
                sr6 = (sigma / r)**6
                e += 4 * epsilon * (sr6*sr6 - sr6)
        # include sidechain-sidechain and sidechain-CA with scaled parameters if present
        if getattr(self.p, 'include_sidechains', False) and self.p.sidechain_coords is not None:
            sc = self.p.sidechain_coords
            for i in range(self.p.N):
                for j in range(i+1, self.p.N):
                    r = distance(sc[i], sc[j])
                    if r > cutoff or r == 0:
                        continue
                    sr6 = ((sigma*0.9) / r)**6
                    e += 4 * (epsilon*0.6) * (sr6*sr6 - sr6)
            # CA - sidechain
            for i in range(self.p.N):
                for j in range(self.p.N):
                    r = distance(self.p.coords[i], sc[j])
                    if r > cutoff or r == 0:
                        continue
                    sr6 = (sigma / r)**6
                    e += 4 * (epsilon*0.5) * (sr6*sr6 - sr6)
        return e

    def hydrophobic_term(self, eps_contact=2.5, r_switch=4.2, r_cut=7.0):
        """Enhanced hydrophobic collapse with increased contact strength.
        
        Increased eps_contact from 0.8 to 2.5 to strengthen hydrophobic driving force.
        """
        import math
        e = 0.0
        r_cut_sq = r_cut * r_cut
        
        # Pre-compute switch function values for common distances
        switch_cache = {}
        def switch(r):
            if r in switch_cache:
                return switch_cache[r]
            if r <= r_switch:
                val = 1.0
            elif r >= r_cut:
                val = 0.0
            else:
                x = (r - r_switch) / (r_cut - r_switch)
                val = 0.5 * (1 + math.cos(math.pi * x))
            switch_cache[r] = val
            return val

        # Fast CA-CA hydrophobic contacts with early cutoff
        coords = self.p.coords
        props = self.p.props
        for i in range(self.p.N):
            ci = coords[i]
            hi = props[i]['hydro']
            if hi < 0.3:  # Skip weakly hydrophobic residues
                continue
            for j in range(i+3, self.p.N):
                cj = coords[j]
                hj = props[j]['hydro']
                if hj < 0.3:  # Skip weakly hydrophobic residues
                    continue
                
                # Fast distance check with squared distance
                dx, dy, dz = ci[0]-cj[0], ci[1]-cj[1], ci[2]-cj[2]
                r_sq = dx*dx + dy*dy + dz*dz
                if r_sq > r_cut_sq:
                    continue
                
                r = math.sqrt(r_sq)
                s = switch(r)
                if s < 0.01:  # Skip negligible contributions
                    continue
                
                # Simplified scaling
                sep = j - i
                scale = 2.5 if sep >= 8 else (2.0 if sep >= 5 else 1.5)
                
                contact_strength = eps_contact * scale * (hi * hj) * s
                if hi > 0.7 and hj > 0.7:
                    contact_strength *= 1.8
                elif hi > 0.5 and hj > 0.5:
                    contact_strength *= 1.4
                
                e += -contact_strength

        return e

    def excluded_volume(self, k_rep=200.0, r_min=2.8):
        """Penalize pairs closer than r_min with a quadratic cost."""
        e = 0.0
        for i in range(self.p.N):
            for j in range(i+2, self.p.N):
                r = distance(self.p.coords[i], self.p.coords[j])
                if r == 0:
                    # extremely strong penalty for overlapping coordinates
                    e += k_rep * 1e3
                    continue
                if r < r_min:
                    e += 0.5 * k_rep * (r_min - r)**2
        return e
    
    def torsion_energy(self, k_torsion=1.0):
        """Calculate proper dihedral angle energies based on Ramachandran preferences.
        
        Reduced k_torsion from 2.0 to 1.0 to prevent excessive torsional strain.
        This computes actual phi/psi-like backbone torsion angles and applies
        sequence-specific preferences to guide folding toward native-like conformations.
        """
        import numpy as np
        if self.p.N < 4:
            return 0.0
        
        e = 0.0
        
        # Calculate proper dihedral angles for each residue
        for i in range(1, self.p.N - 2):
            # Define four consecutive CA atoms for dihedral
            p1 = self.p.coords[i-1]
            p2 = self.p.coords[i]
            p3 = self.p.coords[i+1]
            p4 = self.p.coords[i+2]
            
            # Vectors along bonds
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            
            # Normal vectors to planes
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            
            # Normalize
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm < 1e-6 or n2_norm < 1e-6:
                continue
            
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            
            # Calculate dihedral angle
            cos_dihedral = np.clip(np.dot(n1, n2), -1.0, 1.0)
            dihedral = np.arccos(cos_dihedral)
            
            # Determine sign of dihedral
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            if np.dot(m1, n2) < 0:
                dihedral = -dihedral
            
            # Get amino acid type
            aa = self.p.sequence[i] if i < len(self.p.sequence) else 'A'
            
            # Sequence-specific Ramachandran energy landscape
            # Convert to degrees for easier interpretation
            phi_deg = np.degrees(dihedral)
            
            if aa == 'P':  # Proline - restricted to specific region
                # Proline favors phi ~ -60°
                target_phi = -60.0
                deviation = min(abs(phi_deg - target_phi), abs(phi_deg - target_phi + 360), abs(phi_deg - target_phi - 360))
                e += k_torsion * (deviation / 30.0)**2
                
            elif aa == 'G':  # Glycine - very flexible, allow wide range
                # Glycine can access all regions, minimal penalty
                # Only penalize extreme steric clashes
                if abs(phi_deg) < 30:  # Very restricted region
                    e += k_torsion * 0.5
                    
            elif aa in 'AILMFWYV':  # Hydrophobic - favor beta regions
                # Beta region: phi ~ -120° to -140°
                target_phi = -130.0
                deviation = min(abs(phi_deg - target_phi), abs(phi_deg - target_phi + 360), abs(phi_deg - target_phi - 360))
                if deviation > 60:
                    e += k_torsion * ((deviation - 60) / 40.0)**2
                    
            elif aa in 'DEKR':  # Charged - favor helical regions
                # Alpha helix region: phi ~ -60° to -70°
                target_phi = -65.0
                deviation = min(abs(phi_deg - target_phi), abs(phi_deg - target_phi + 360), abs(phi_deg - target_phi - 360))
                if deviation > 50:
                    e += k_torsion * ((deviation - 50) / 35.0)**2
                    
            else:  # Other residues - balanced preferences
                # Favor allowed regions of Ramachandran plot
                # Penalize positive phi angles (generally forbidden)
                if phi_deg > 0 and phi_deg < 120:
                    e += k_torsion * 2.0 * (phi_deg / 60.0)**2
        
        return e
    
    def contact_guidance_term(self, predicted_contacts=None, k_contact=1.5):
        """Apply soft constraints based on predicted or reference contacts.
        
        Args:
            predicted_contacts: List of tuples (i, j, probability) where
                i, j are residue indices and probability is confidence (0-1)
            k_contact: Force constant for contact constraints
        
        Returns:
            Energy value (negative for satisfied contacts, positive for violations)
        """
        import numpy as np
        
        if predicted_contacts is None or len(predicted_contacts) == 0:
            return 0.0
        
        e = 0.0
        coords = self.p.coords
        
        for contact in predicted_contacts:
            if len(contact) == 3:
                i, j, prob = contact
            else:
                i, j = contact[:2]
                prob = 1.0  # Default confidence
            
            # Skip invalid indices
            if i >= self.p.N or j >= self.p.N or i < 0 or j < 0:
                continue
            
            # Skip nearby residues (not informative)
            if abs(i - j) < 4:
                continue
            
            # Calculate distance
            r = distance(coords[i], coords[j])
            
            # Contact is satisfied if distance < 8 Angstroms
            contact_distance = 8.0
            
            if prob > 0.5:  # High confidence contact
                if r < contact_distance:
                    # Reward contact formation (negative energy)
                    strength = k_contact * prob * (1.0 - r / contact_distance)
                    e += -strength * 2.0
                else:
                    # Penalize missing high-confidence contact
                    violation = (r - contact_distance) / contact_distance
                    e += k_contact * prob * violation**2
                    
            elif prob > 0.2:  # Medium confidence contact
                if r < contact_distance:
                    # Mild reward for contact
                    strength = k_contact * prob * (1.0 - r / contact_distance)
                    e += -strength
                else:
                    # Mild penalty for missing contact
                    violation = (r - contact_distance) / contact_distance
                    e += 0.5 * k_contact * prob * violation**2
            
            # Low confidence contacts (prob < 0.2) are ignored
        
        return e

    def electrostatic(self, k_e=4.0, dielectric=10.0):
        # default arguments here are placeholders; prefer calling with real constants
        e = 0.0
        for i in range(self.p.N):
            for j in range(i+2, self.p.N):
                q1 = self.p.props[i]['charge']
                q2 = self.p.props[j]['charge']
                if q1 == 0 or q2 == 0:
                    continue
                r = distance(self.p.coords[i], self.p.coords[j])
                if r == 0:
                    # strong penalty for overlapping charges
                    e += 1e3
                    continue
                e += k_e * (q1 * q2) / (dielectric * r)
        return e

    def total_energy(self, predicted_contacts=None):
        """Calculate total energy with balanced force constants.
        
        Parameters optimized for realistic protein folding while avoiding
        excessive strain from overly strong harmonic potentials.
        
        Args:
            predicted_contacts: Optional list of (i, j, prob) tuples for contact guidance
        """
        K_COUL = 332.06371
        DIELECTRIC = 80.0
        
        # Core geometric constraints (reduced to prevent dominance)
        e_bond = self.bond_energy(k_bond=50.0)  # Standard bond strength
        e_angle = self.angle_energy(k_angle=4.0)  # Further reduced - more flexibility
        e_excl = self.excluded_volume(k_rep=200.0, r_min=2.8)  # Standard repulsion
        
        # Non-bonded interactions (enhanced for better folding)
        e_lj = self.lj_nonbonded(epsilon=0.2, sigma=3.8)  # Increased LJ attraction
        e_hydro = self.hydrophobic_term(eps_contact=2.5)  # Strongly enhanced hydrophobic collapse
        e_elec = self.electrostatic(k_e=K_COUL, dielectric=DIELECTRIC)
        
        # Structure-guiding terms (enhanced for stability)
        e_compact = self.compactness_bias(k_compact=1.0)  # Increased compactness drive
        e_native = self.native_bias(k_native=0.5)  # Ramachandran preferences
        e_ss = self.secondary_structure_bias(k_ss=1.5)  # Enhanced secondary structure
        e_hbond = self.hydrogen_bond_term(k_hbond=2.5)  # Significantly enhanced H-bonding
        
        # Ramachandran-based dihedral restraints (NEW - proper phi/psi statistics)
        e_ramachandran = self.ramachandran_restraints(k_rama=1.5)  # Statistical backbone preferences
        
        # Legacy torsional terms (reduced weight since Ramachandran is more accurate)
        e_torsion_proper = self.torsion_energy(k_torsion=0.5)  # Reduced - Ramachandran handles this better
        e_torsion_pseudo = self.torsional_bias(k_torsion=0.1)  # Further reduced weight
        
        e_loop = self.loop_closure_bias(k_loop=0.6)  # Loop geometry
        
        # Optional contact guidance (if AlphaFold predictions available)
        e_contact = 0.0
        if predicted_contacts is not None:
            e_contact = self.contact_guidance_term(predicted_contacts, k_contact=1.0)
        
        # Total energy
        total_E = (e_bond + e_angle + e_lj + e_hydro + e_elec + e_excl + 
                   e_compact + e_native + e_ss + e_hbond + e_ramachandran +
                   e_torsion_proper + e_torsion_pseudo + e_loop + e_contact)

        return {
            'bond': e_bond,
            'angle': e_angle,
            'lj': e_lj,
            'hydro': e_hydro,
            'elec': e_elec,
            'excl': e_excl,
            'compact': e_compact,
            'native': e_native,
            'ss': e_ss,
            'hbond': e_hbond,
            'ramachandran': e_ramachandran,  # NEW: Proper Ramachandran restraints
            'torsion': e_torsion_proper + e_torsion_pseudo,  # Combined legacy torsion
            'loop': e_loop,
            'contact': e_contact,
            'total': total_E
        }
    
    def compactness_bias(self, k_compact=0.5):
        """Enhanced compactness bias with sequence-length scaling for maximum accuracy"""
        import numpy as np
        coords = self.p.coords
        centroid = np.mean(coords, axis=0)
        rg_sq = np.mean(np.sum((coords - centroid)**2, axis=1))
        
        # Sequence-length dependent ideal radius of gyration
        # Empirical scaling: Rg ~ N^0.6 for folded proteins
        N = self.p.N
        ideal_rg = 2.2 * (N ** 0.6)  # Empirical constant for folded proteins
        
        # Penalty for deviation from ideal compactness
        rg = np.sqrt(rg_sq)
        deviation = abs(rg - ideal_rg) / ideal_rg
        
        # Stronger penalty for overly extended structures
        if rg > ideal_rg * 1.5:
            penalty = k_compact * 3.0 * deviation**2
        elif rg > ideal_rg * 1.2:
            penalty = k_compact * 2.0 * deviation**2
        else:
            penalty = k_compact * deviation**2
        
        return penalty
    
    def native_bias(self, k_native=1.0):
        """Ultra-enhanced bias toward native-like structures with Ramachandran preferences"""
        import numpy as np
        if self.p.N < 4:
            return 0.0
        
        e = 0.0
        
        # Enhanced dihedral angle bias with sequence-specific preferences
        for i in range(1, self.p.N - 2):
            # Calculate pseudo-dihedral angle
            v1 = self.p.coords[i] - self.p.coords[i-1]
            v2 = self.p.coords[i+1] - self.p.coords[i]
            v3 = self.p.coords[i+2] - self.p.coords[i+1]
            
            # Cross products for dihedral calculation
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            
            # Normalize
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm > 1e-6 and n2_norm > 1e-6:
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm
                
                # Dihedral angle
                cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
                phi = np.arccos(cos_phi)
                
                # Get amino acid type for sequence-specific bias
                aa = self.p.sequence[i] if i < len(self.p.sequence) else 'A'
                
                # Sequence-specific Ramachandran preferences
                if aa == 'P':  # Proline - restricted phi angles
                    # Favor phi around -60 degrees (cis-proline region)
                    target_phi = np.pi * 2/3  # ~120 degrees
                    e += -k_native * 2.0 * np.exp(-(phi - target_phi)**2 / 0.5)
                elif aa == 'G':  # Glycine - more flexible
                    # Allow wider range but still favor reasonable angles
                    if np.pi/4 <= phi <= 3*np.pi/4:
                        e += -k_native * 0.8
                elif aa in 'AILMFWYV':  # Hydrophobic - favor extended/beta
                    # Favor extended conformations (phi ~ 120-140 degrees)
                    target_phi = np.pi * 2.3/3  # ~138 degrees
                    e += -k_native * 1.5 * np.exp(-(phi - target_phi)**2 / 0.8)
                elif aa in 'DEKR':  # Charged - favor helical
                    # Favor alpha-helical conformations (phi ~ 100 degrees)
                    target_phi = np.pi * 1.8/3  # ~108 degrees
                    e += -k_native * 1.8 * np.exp(-(phi - target_phi)**2 / 0.6)
                else:  # Others - balanced preference
                    # Favor reasonable phi angles (avoid extreme values)
                    if np.pi/3 <= phi <= 2*np.pi/3:
                        e += -k_native * 1.2
                    elif phi < np.pi/6 or phi > 5*np.pi/6:
                        e += k_native * 2.0 * (np.pi/3 - min(phi, np.pi - phi))**2
        
        # Enhanced local geometry bias
        for i in range(2, self.p.N - 1):
            # Favor reasonable CA-CA-CA angles (close to tetrahedral)
            v1 = self.p.coords[i-1] - self.p.coords[i]
            v2 = self.p.coords[i+1] - self.p.coords[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Favor angles around 109.5 degrees (tetrahedral)
                target_angle = np.pi * 109.5 / 180.0
                e += -k_native * 0.8 * np.exp(-(angle - target_angle)**2 / 0.3)
        
        return e
    
    def secondary_structure_bias(self, k_ss=0.8):
        """Fast secondary structure bias with distance cutoffs"""
        import numpy as np
        if self.p.N < 6:
            return 0.0
        
        e = 0.0
        coords = self.p.coords
        
        # Fast alpha-helix bias (i, i+4 contacts)
        for i in range(self.p.N - 4):
            ci, cj = coords[i], coords[i+4]
            dx, dy, dz = ci[0]-cj[0], ci[1]-cj[1], ci[2]-cj[2]
            r_sq = dx*dx + dy*dy + dz*dz
            if 25.0 <= r_sq <= 64.0:  # 5.0² to 8.0²
                r = np.sqrt(r_sq)
                e += -k_ss * 2.5 * np.exp(-(r - 6.2)**2 / 1.2)
        
        # Fast beta-sheet bias (i, i+3 contacts)
        for i in range(self.p.N - 3):
            ci, cj = coords[i], coords[i+3]
            dx, dy, dz = ci[0]-cj[0], ci[1]-cj[1], ci[2]-cj[2]
            r_sq = dx*dx + dy*dy + dz*dz
            if 20.25 <= r_sq <= 42.25:  # 4.5² to 6.5²
                r = np.sqrt(r_sq)
                e += -k_ss * 1.8 * np.exp(-(r - 5.5)**2 / 0.8)
        
        # Simplified local contacts (reduced range for speed)
        for i in range(self.p.N - 3):
            for j in range(i + 3, min(i + 6, self.p.N)):  # Reduced from i+7
                ci, cj = coords[i], coords[j]
                dx, dy, dz = ci[0]-cj[0], ci[1]-cj[1], ci[2]-cj[2]
                r_sq = dx*dx + dy*dy + dz*dz
                if 6.25 <= r_sq <= 20.25:  # 2.5² to 4.5²
                    r = np.sqrt(r_sq)
                    e += -k_ss * 1.5 * np.exp(-(r - 3.2)**2 / 0.8)
        
        return e
    
    def torsional_bias(self, k_torsion=0.5):
        """Ultra-enhanced torsional bias with Ramachandran-like preferences for maximum accuracy"""
        import numpy as np
        if self.p.N < 4:
            return 0.0
        
        e = 0.0
        
        # Enhanced pseudo-dihedral calculations with sequence-specific preferences
        for i in range(1, self.p.N - 2):
            # Calculate pseudo-torsion angle (phi-like)
            v1 = self.p.coords[i-1] - self.p.coords[i]
            v2 = self.p.coords[i] - self.p.coords[i+1]
            v3 = self.p.coords[i+1] - self.p.coords[i+2]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            v3_norm = np.linalg.norm(v3)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6 and v3_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                v3 = v3 / v3_norm
                
                # Calculate consecutive bond angles
                angle1 = np.arccos(np.clip(np.dot(-v1, v2), -1.0, 1.0))
                angle2 = np.arccos(np.clip(np.dot(-v2, v3), -1.0, 1.0))
                
                # Get amino acid for sequence-specific preferences
                aa = self.p.sequence[i] if i < len(self.p.sequence) else 'A'
                
                # Sequence-specific angle preferences (Ramachandran-like)
                if aa == 'P':  # Proline - very restricted
                    # Proline favors specific angles due to ring constraint
                    target1 = np.pi * 100.0 / 180.0  # ~100°
                    target2 = np.pi * 110.0 / 180.0  # ~110°
                    e += -k_torsion * 3.0 * (np.exp(-(angle1 - target1)**2 / 0.2) + 
                                             np.exp(-(angle2 - target2)**2 / 0.2))
                elif aa == 'G':  # Glycine - very flexible
                    # Glycine allows wider range but still favor reasonable angles
                    if np.pi/4 <= angle1 <= 3*np.pi/4 and np.pi/4 <= angle2 <= 3*np.pi/4:
                        e += -k_torsion * 1.0
                elif aa in 'AILMFWYV':  # Hydrophobic - favor extended/beta
                    # Beta-strand like angles
                    target1 = np.pi * 120.0 / 180.0  # ~120°
                    target2 = np.pi * 120.0 / 180.0  # ~120°
                    e += -k_torsion * 2.0 * (np.exp(-(angle1 - target1)**2 / 0.4) + 
                                             np.exp(-(angle2 - target2)**2 / 0.4))
                elif aa in 'DEKRHQN':  # Polar/charged - favor helical
                    # Alpha-helix like angles
                    target1 = np.pi * 105.0 / 180.0  # ~105°
                    target2 = np.pi * 115.0 / 180.0  # ~115°
                    e += -k_torsion * 2.2 * (np.exp(-(angle1 - target1)**2 / 0.3) + 
                                             np.exp(-(angle2 - target2)**2 / 0.3))
                else:  # Others - balanced
                    # Favor tetrahedral-like angles
                    target_angle = np.pi * 109.5 / 180.0
                    e += -k_torsion * 1.5 * (np.exp(-(angle1 - target_angle)**2 / 0.5) + 
                                             np.exp(-(angle2 - target_angle)**2 / 0.5))
                
                # Universal penalty for extreme angles (avoid steric clashes)
                if angle1 < np.pi/8 or angle1 > 7*np.pi/8:
                    e += k_torsion * 4.0 * (np.pi/4 - min(angle1, np.pi - angle1))**2
                if angle2 < np.pi/8 or angle2 > 7*np.pi/8:
                    e += k_torsion * 4.0 * (np.pi/4 - min(angle2, np.pi - angle2))**2
        
        # Additional pseudo-psi angle bias
        for i in range(self.p.N - 3):
            # Calculate another pseudo-torsion (psi-like)
            if i + 3 < self.p.N:
                v1 = self.p.coords[i] - self.p.coords[i+1]
                v2 = self.p.coords[i+1] - self.p.coords[i+2]
                v3 = self.p.coords[i+2] - self.p.coords[i+3]
                
                # Cross products for dihedral-like calculation
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                
                n1_norm = np.linalg.norm(n1)
                n2_norm = np.linalg.norm(n2)
                
                if n1_norm > 1e-6 and n2_norm > 1e-6:
                    n1 = n1 / n1_norm
                    n2 = n2 / n2_norm
                    
                    cos_dihedral = np.clip(np.dot(n1, n2), -1.0, 1.0)
                    dihedral = np.arccos(cos_dihedral)
                    
                    # Favor dihedral angles in allowed regions
                    aa = self.p.sequence[i+1] if i+1 < len(self.p.sequence) else 'A'
                    
                    if aa in 'AILMFWYV':  # Beta-favoring
                        # Favor extended conformations
                        if np.pi/2 <= dihedral <= np.pi:
                            e += -k_torsion * 1.0
                    elif aa in 'DEKRHQN':  # Helix-favoring
                        # Favor helical conformations
                        if np.pi/4 <= dihedral <= 3*np.pi/4:
                            e += -k_torsion * 1.2
        
        return e
    
    def hydrogen_bond_term(self, k_hbond=2.5):
        """Enhanced hydrogen bonding to stabilize secondary structures.
        
        Increased k_hbond from 1.0 to 2.5 to strengthen H-bond contributions.
        """
        import numpy as np
        if self.p.N < 4:
            return 0.0
        
        e = 0.0
        
        # Fast donor/acceptor identification
        donors = [i for i, aa in enumerate(self.p.sequence) if aa in 'NQSTYW']
        acceptors = [i for i, aa in enumerate(self.p.sequence) if aa in 'DEQNSTY']
        
        if not donors or not acceptors:
            return 0.0
        
        coords = self.p.coords
        sequence = self.p.sequence
        
        # Fast H-bond calculation with distance cutoff
        for donor in donors:
            cd = coords[donor]
            for acceptor in acceptors:
                sep = abs(donor - acceptor)
                if sep < 3:  # Skip nearby residues
                    continue
                
                ca = coords[acceptor]
                dx, dy, dz = cd[0]-ca[0], cd[1]-ca[1], cd[2]-ca[2]
                r_sq = dx*dx + dy*dy + dz*dz
                
                if 5.3 <= r_sq <= 16.0:  # 2.3² to 4.0²
                    r = np.sqrt(r_sq)
                    
                    # Fast separation bonus
                    sep_bonus = 2.5 if sep >= 8 else (2.0 if sep >= 5 else 1.5)
                    
                    # Simplified H-bond potential
                    if r <= 2.9:
                        hb_energy = -k_hbond * sep_bonus * (2.9/r)**6
                    else:
                        hb_energy = -k_hbond * sep_bonus * np.exp(-(r - 2.9)**2 / 0.5)
                    
                    # Fast AA bonus lookup
                    donor_aa = sequence[donor]
                    acceptor_aa = sequence[acceptor]
                    if (donor_aa in 'NQ' and acceptor_aa in 'DE'):
                        hb_energy *= 1.6
                    elif donor_aa in 'NQ' and acceptor_aa in 'NQ':
                        hb_energy *= 1.4
                    
                    e += hb_energy
        
        return e
    
    def loop_closure_bias(self, k_loop=0.8):
        """Fast loop closure with reduced complexity"""
        import numpy as np
        if self.p.N < 6:
            return 0.0
        
        e = 0.0
        coords = self.p.coords
        
        # Simplified loop configs (fewer sizes for speed)
        loop_configs = [(4, 6.5, 1.2), (6, 8.5, 1.6), (8, 10.5, 2.0), (10, 12.5, 2.4)]
        
        for loop_size, ideal_r, tolerance in loop_configs:
            if self.p.N <= loop_size:
                continue
                
            for i in range(self.p.N - loop_size):
                j = i + loop_size
                ci, cj = coords[i], coords[j]
                dx, dy, dz = ci[0]-cj[0], ci[1]-cj[1], ci[2]-cj[2]
                r_sq = dx*dx + dy*dy + dz*dz
                
                if r_sq < (ideal_r * 2.0)**2:
                    r = np.sqrt(r_sq)
                    strength = k_loop * (1.0 + 0.2 * loop_size / 15.0)
                    e += -strength * np.exp(-(r - ideal_r)**2 / (2.0 * tolerance**2))
        
        # Fast turn geometry (reduced complexity)
        for i in range(self.p.N - 4):
            ci, c3, c4 = coords[i], coords[i+3], coords[i+4]
            
            # i, i+3 turn
            dx, dy, dz = ci[0]-c3[0], ci[1]-c3[1], ci[2]-c3[2]
            r3_sq = dx*dx + dy*dy + dz*dz
            if 16.0 <= r3_sq <= 49.0:  # 4.0² to 7.0²
                r3 = np.sqrt(r3_sq)
                e += -k_loop * 1.2 * np.exp(-(r3 - 5.5)**2 / 1.0)
            
            # i, i+4 turn
            dx, dy, dz = ci[0]-c4[0], ci[1]-c4[1], ci[2]-c4[2]
            r4_sq = dx*dx + dy*dy + dz*dz
            if 25.0 <= r4_sq <= 64.0:  # 5.0² to 8.0²
                r4 = np.sqrt(r4_sq)
                e += -k_loop * 1.0 * np.exp(-(r4 - 6.5)**2 / 1.2)
        
        return e

    # ========== Ramachandran Restraints Implementation ==========
    
    def _init_ramachandran_probs(self):
        """Initialize Ramachandran probability distributions for amino acids.
        
        These are simplified probability distributions based on observed phi/psi angles
        from high-resolution crystal structures. Probabilities are binned into 10-degree
        intervals for efficient lookup.
        """
        # Simplified Ramachandran regions (phi, psi, probability)
        # Based on Lovell et al. (2003) and PISCES database
        
        self.rama_distributions = {
            'G': {  # Glycine - very flexible, accesses all regions
                'alpha_R': (-60, -45, 0.35),  # Right-handed alpha helix
                'alpha_L': (60, 45, 0.25),     # Left-handed alpha (allowed for Gly)
                'beta': (-120, 120, 0.25),     # Beta sheet
                'other': (0, 0, 0.15)          # Other allowed regions
            },
            'P': {  # Proline - very restricted due to ring
                'polyproline': (-60, 150, 0.70),  # Polyproline II
                'cis': (-60, -30, 0.20),          # Cis-proline
                'other': (-90, 0, 0.10)
            },
            'pre_P': {  # Residue before proline (more flexible)
                'extended': (-60, 150, 0.50),
                'alpha': (-60, -45, 0.30),
                'other': (0, 0, 0.20)
            },
            'IVL': {  # Branched aliphatic - slightly restricted
                'alpha_R': (-60, -45, 0.45),
                'beta': (-120, 120, 0.40),
                'other': (-90, 0, 0.15)
            },
            'General': {  # All other amino acids
                'alpha_R': (-60, -45, 0.40),    # Right-handed alpha helix
                'beta': (-120, 120, 0.35),      # Beta sheet
                'left_alpha': (60, 45, 0.05),   # Left-handed alpha (rare)
                'other': (-90, 0, 0.20)         # Other allowed
            }
        }
    
    def _get_rama_type(self, aa, position, sequence_length):
        """Determine Ramachandran distribution type for amino acid."""
        if aa == 'G':
            return 'G'
        elif aa == 'P':
            return 'P'
        elif position < sequence_length - 1 and self.p.sequence[position + 1] == 'P':
            return 'pre_P'
        elif aa in 'IVL':
            return 'IVL'
        else:
            return 'General'
    
    def _calculate_dihedral(self, p1, p2, p3, p4):
        """Calculate dihedral angle between four points.
        
        Args:
            p1, p2, p3, p4: 3D coordinate arrays
            
        Returns:
            Dihedral angle in radians (-π to π)
        """
        # Vectors along bonds
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Normal vectors to planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Check for degenerate cases
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return 0.0
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Calculate dihedral using atan2 for proper quadrant
        b2_unit = b2 / np.linalg.norm(b2)
        m1 = np.cross(n1, b2_unit)
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        return np.arctan2(y, x)
    
    def _approximate_backbone_atoms(self, i):
        """Approximate N and C positions from C-alpha coordinates.
        
        For a C-alpha-only model, we approximate the backbone N and C atoms
        using idealized geometry:
        - N-CA bond: 1.46 Å
        - CA-C bond: 1.52 Å
        - Bond angles: ~110°
        
        Args:
            i: Residue index
            
        Returns:
            tuple: (N_pos, C_pos, N_next_pos) as numpy arrays
        """
        if i < 1 or i >= self.p.N - 1:
            return None, None, None
        
        CA_prev = self.p.coords[i-1]
        CA_curr = self.p.coords[i]
        CA_next = self.p.coords[i+1]
        
        # Direction vectors
        vec_prev = CA_curr - CA_prev
        vec_next = CA_next - CA_curr
        
        vec_prev_norm = np.linalg.norm(vec_prev)
        vec_next_norm = np.linalg.norm(vec_next)
        
        if vec_prev_norm < 1e-6 or vec_next_norm < 1e-6:
            return None, None, None
        
        vec_prev_unit = vec_prev / vec_prev_norm
        vec_next_unit = vec_next / vec_next_norm
        
        # Approximate N position (1.46 Å toward previous CA)
        N_pos = CA_curr - 1.46 * vec_prev_unit
        
        # Approximate C position (1.52 Å toward next CA)
        C_pos = CA_curr + 1.52 * vec_next_unit
        
        # Approximate next N position
        if i + 2 < self.p.N:
            CA_next2 = self.p.coords[i+2]
            vec_next2 = CA_next2 - CA_next
            vec_next2_norm = np.linalg.norm(vec_next2)
            if vec_next2_norm > 1e-6:
                vec_next2_unit = vec_next2 / vec_next2_norm
                N_next_pos = CA_next - 1.46 * vec_next_unit
            else:
                N_next_pos = CA_next - 1.46 * vec_next_unit
        else:
            N_next_pos = CA_next - 1.46 * vec_next_unit
        
        return N_pos, C_pos, N_next_pos
    
    def _evaluate_rama_energy(self, phi, psi, aa, position):
        """Evaluate Ramachandran energy for given phi/psi angles.
        
        Args:
            phi: Phi angle in radians
            psi: Psi angle in radians
            aa: Amino acid one-letter code
            position: Residue position in sequence
            
        Returns:
            Energy value (lower for favored regions)
        """
        # Convert to degrees for comparison
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)
        
        # Get appropriate distribution
        rama_type = self._get_rama_type(aa, position, self.p.N)
        distribution = self.rama_distributions.get(rama_type, self.rama_distributions['General'])
        
        # Calculate probability based on distance to favored regions
        max_prob = 0.0
        
        for region_name, (phi_center, psi_center, prob_weight) in distribution.items():
            # Angular distance (handling wraparound at ±180°)
            dphi = abs(phi_deg - phi_center)
            dpsi = abs(psi_deg - psi_center)
            
            # Handle wraparound
            if dphi > 180:
                dphi = 360 - dphi
            if dpsi > 180:
                dpsi = 360 - dpsi
            
            # Calculate probability using Gaussian-like distribution
            # Sigma = 30° for most regions, 40° for flexible regions
            sigma = 40.0 if rama_type == 'G' else 30.0
            
            angular_dist_sq = (dphi**2 + dpsi**2) / (2 * sigma**2)
            region_prob = prob_weight * np.exp(-angular_dist_sq)
            
            max_prob = max(max_prob, region_prob)
        
        # Convert probability to energy: E = -kT * ln(P)
        # Use kT ~ 0.6 kcal/mol at room temperature
        if max_prob > 1e-6:
            energy = -0.6 * np.log(max_prob)
        else:
            # Heavy penalty for strongly forbidden regions
            energy = 10.0
        
        return energy
    
    def ramachandran_restraints(self, k_rama=1.5):
        """Apply Ramachandran-based dihedral angle restraints.
        
        This function calculates pseudo-phi and pseudo-psi dihedral angles from
        C-alpha coordinates and applies energy penalties based on Ramachandran
        probability distributions. This guides the folding toward realistic
        backbone conformations.
        
        Args:
            k_rama: Force constant for Ramachandran restraints
            
        Returns:
            Total Ramachandran energy
        """
        if self.p.N < 4:
            return 0.0
        
        e = 0.0
        
        # Calculate dihedral angles for each residue
        for i in range(1, self.p.N - 2):
            # Approximate backbone atoms from C-alpha positions
            N_pos, C_pos, N_next_pos = self._approximate_backbone_atoms(i)
            
            if N_pos is None:
                continue
            
            # Need C from previous residue for phi
            if i > 1:
                _, C_prev_pos, _ = self._approximate_backbone_atoms(i-1)
                if C_prev_pos is not None:
                    # Calculate phi: C(i-1) - N(i) - CA(i) - C(i)
                    phi = self._calculate_dihedral(C_prev_pos, N_pos, 
                                                   self.p.coords[i], C_pos)
                else:
                    phi = 0.0
            else:
                phi = 0.0
            
            # Calculate psi: N(i) - CA(i) - C(i) - N(i+1)
            psi = self._calculate_dihedral(N_pos, self.p.coords[i], 
                                          C_pos, N_next_pos)
            
            # Get amino acid type
            aa = self.p.sequence[i] if i < len(self.p.sequence) else 'A'
            
            # Evaluate Ramachandran energy
            rama_energy = self._evaluate_rama_energy(phi, psi, aa, i)
            
            e += k_rama * rama_energy
        
        return e

__all__ = ['EnergyFunction']
