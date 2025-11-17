from .models import distance


class EnergyFunction:
    def __init__(self, protein):
        self.p = protein

    def bond_energy(self, k_bond=50.0, r0=3.8):
        if r0 is None:
            r0 = self.p.ca_distance
        e = 0.0
        for i in range(self.p.N - 1):
            r = distance(self.p.coords[i], self.p.coords[i+1])
            e += 0.5 * k_bond * (r - r0)**2
        return e

    def angle_energy(self, k_angle=20.0, theta0=None):
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

    def hydrophobic_term(self, eps_contact=0.8, r_switch=6.0, r_cut=8.0):
        """Contact-like hydrophobic attraction with a smooth switching function.

        eps_contact: kcal/mol scaling for maximal contact
        r_switch: distance below which contact is fully on
        r_cut: distance above which contact is zero
        """
        import math
        e = 0.0

        def switch(r):
            if r <= r_switch:
                return 1.0
            if r >= r_cut:
                return 0.0
            # smooth cosine switch between r_switch and r_cut
            x = (r - r_switch) / (r_cut - r_switch)
            return 0.5 * (1 + math.cos(math.pi * x))

        # CA-CA contacts
        for i in range(self.p.N):
            for j in range(i+2, self.p.N):
                r = distance(self.p.coords[i], self.p.coords[j])
                if r == 0:
                    continue
                hi = self.p.props[i]['hydro']
                hj = self.p.props[j]['hydro']
                s = switch(r)
                e += -eps_contact * (hi * hj) * s

        # sidechain contacts if present (stronger magnitude)
        if getattr(self.p, 'include_sidechains', False) and self.p.sidechain_coords is not None:
            sc = self.p.sidechain_coords
            for i in range(self.p.N):
                for j in range(i+1, self.p.N):
                    r = distance(sc[i], sc[j])
                    if r == 0:
                        continue
                    hi = self.p.props[i]['hydro']
                    hj = self.p.props[j]['hydro']
                    s = switch(r)
                    e += -1.2 * eps_contact * (hi * hj) * s

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

    def total_energy(self):
        # Use physical-inspired parameters (units ~ kcal/mol, lengths in Angstroms)
        # Coulomb prefactor in kcal·Å/mol·e^2
        K_COUL = 332.06371
        DIELECTRIC = 80.0
        e_bond = self.bond_energy(k_bond=300.0)
        e_angle = self.angle_energy(k_angle=30.0)
        e_lj = self.lj_nonbonded(epsilon=0.3, sigma=3.8)
        # hydrophobic: normalize hydro values roughly to 0..1 and multiply by contact eps
        e_hydro = self.hydrophobic_term(eps_contact=0.8)
        e_elec = self.electrostatic(k_e=K_COUL, dielectric=DIELECTRIC)
        e_excl = self.excluded_volume(k_rep=500.0, r_min=3.4)

        total_E = e_bond + e_angle + e_lj + e_hydro + e_elec + e_excl

        return {
            'bond': e_bond,
            'angle': e_angle,
            'lj': e_lj,
            'hydro': e_hydro,
            'elec': e_elec,
            'excl': e_excl,
            'total': total_E
        }

__all__ = ['EnergyFunction']
