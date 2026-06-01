import pandas as pd
import numpy as np
from config import NEPRA_TARIFF_SLABS, PROTECTED_STATUS_THRESHOLD


class BillingEngine:
    def __init__(self, config):
        self.config = config
        self.tariff_slabs = NEPRA_TARIFF_SLABS
        
    def calculate_slab(self, units):
        """Determine which tariff slab the units fall into"""
        if units <= 100:
            return "0-100", self.tariff_slabs["0-100"]
        elif units <= 200:
            return "101-200", self.tariff_slabs["101-200"]
        elif units <= 300:
            return "201-300", self.tariff_slabs["201-300"]
        elif units <= 500:
            return "301-500", self.tariff_slabs["301-500"]
        else:
            return "500+", self.tariff_slabs["500+"]
    
    def calculate_bill(self, units, previous_bill=None):
        """Calculate total bill amount"""
        slab_range, slab_info = self.calculate_slab(units)
        
        base_charge = 300
        unit_cost = units * slab_info['rate']
        taxes = unit_cost * 0.17
        
        total = base_charge + unit_cost + taxes
        
        return {
            'units': units,
            'slab': slab_range,
            'rate_per_unit': slab_info['rate'],
            'base_charge': base_charge,
            'unit_cost': unit_cost,
            'taxes': taxes,
            'total_bill': total,
            'category': slab_info['category'],
            'is_protected': units <= PROTECTED_STATUS_THRESHOLD
        }
    
    def calculate_savings_target(self, current_units, target_slab="0-100"):
        """Calculate how many units to cut to reach target slab"""
        target_limits = {
            "0-100": 100,
            "101-200": 200,
            "201-300": 300,
            "301-500": 500
        }
        
        target_limit = target_limits.get(target_slab, 100)
        
        if current_units <= target_limit:
            return {
                'current_units': current_units,
                'target_units': target_limit,
                'units_to_cut': 0,
                'already_in_target': True,
                'potential_savings': 0
            }
        
        units_to_cut = current_units - target_limit
        current_bill = self.calculate_bill(current_units)['total_bill']
        target_bill = self.calculate_bill(target_limit)['total_bill']
        
        return {
            'current_units': current_units,
            'target_units': target_limit,
            'units_to_cut': units_to_cut,
            'percent_reduction': (units_to_cut / current_units) * 100,
            'already_in_target': False,
            'current_bill': current_bill,
            'target_bill': target_bill,
            'potential_savings': current_bill - target_bill,
            'savings_percent': ((current_bill - target_bill) / current_bill) * 100
        }
    
    def apply_differential_tariff(self, units_on_peak, units_off_peak):
        """Calculate bill with peak/off-peak differentiation"""
        peak_rate = 28.0
        off_peak_rate = 16.0
        
        peak_cost = units_on_peak * peak_rate
        off_peak_cost = units_off_peak * off_peak_rate
        
        total_units = units_on_peak + units_off_peak
        total_cost = peak_cost + off_peak_cost
        
        base_charge = 300
        taxes = total_cost * 0.17
        
        return {
            'units_on_peak': units_on_peak,
            'units_off_peak': units_off_peak,
            'total_units': total_units,
            'peak_cost': peak_cost,
            'off_peak_cost': off_peak_cost,
            'base_charge': base_charge,
            'taxes': taxes,
            'total_bill': base_charge + total_cost + taxes,
            'effective_rate': (total_cost / total_units) if total_units > 0 else 0
        }
    
    def get_bill_comparison(self, units):
        """Compare bills across different scenarios"""
        standard_bill = self.calculate_bill(units)
        
        peak_off_peak_split = int(units * 0.3)
        differential_bill = self.apply_differential_tariff(
            peak_off_peak_split, 
            units - peak_off_peak_split
        )
        
        return {
            'standard': standard_bill,
            'differential': differential_bill,
            'savings_with_shifting': standard_bill['total_bill'] - differential_bill['total_bill']
        }
    
    def generate_bill_breakdown(self, units):
        """Generate detailed bill breakdown"""
        bill = self.calculate_bill(units)
        
        return pd.DataFrame({
            'Component': ['Base Charge', 'Unit Cost', 'Taxes', 'Total Bill'],
            'Amount (PKR)': [
                bill['base_charge'],
                bill['unit_cost'],
                bill['taxes'],
                bill['total_bill']
            ],
            'Percentage': [
                (bill['base_charge'] / bill['total_bill']) * 100,
                (bill['unit_cost'] / bill['total_bill']) * 100,
                (bill['taxes'] / bill['total_bill']) * 100,
                100
            ]
        })
