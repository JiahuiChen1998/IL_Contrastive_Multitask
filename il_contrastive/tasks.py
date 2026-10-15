"""Explicit workbook schema. Native units stay explicit when not verified."""
def task(file, sheet, target, unit, temperature=None, pressure=None, pressure_factor=1.,
         pressure_unit=None, transform='identity', head='mlp'):
    return dict(file=file, sheet=sheet, target=target, unit=unit,
                temperature=temperature, pressure=pressure, pressure_factor=pressure_factor,
                pressure_unit=pressure_unit, transform=transform, head=head)

TASKS = {
 'melting_point': task('melting point.xlsx', 'Sheet1', 'MP_K', 'K'),
 'glass_transition': task('GTT_V.xlsx', 'Sheet1', 'GTT', 'degC'),
 'decomposition_temperature': task('TDECOMP_V.xlsx', 'Sheet1', 'TDECOMP_Celcius', 'degC'),
 'toxicity': task('IL-TOX.xlsx', 'TOX', 'Experimental logEC50', 'log10(EC50), source convention'),
 'density': task('density_P.xlsx', 'raw', 'ρ / kg/m3', 'kg/m3', 'T / K', 'p / MPa', pressure_unit='MPa'),
 'viscosity': task('viscosity_P.xlsx', 'viscosity', 'η / mPa s', 'mPa s', 'T / K', transform='log', head='vft'),
 'heat_capacity': task('heat capacity_V.xlsx', 'Sheet1', 'HCAP', 'workbook HCAP unit (unverified)',
                       'Temperature', 'Pressure', pressure_unit='workbook Pressure unit (unverified)', transform='log'),
 'refractive_index': task('RI_V.xlsx', 'Sheet1', 'RI', 'dimensionless', 'TEMPERATURE(K)'),
 'surface_tension': task('SURFACETENSION_V.xlsx', 'Sheet1', 'SurfaceTension', 'workbook native (consistent with N/m)',
                         'Temperature', head='linear'),
 'thermal_conductivity': task('THCOND_V.xlsx', 'Sheet1', 'THCOND', 'workbook native (consistent with W/(m K))', 'Temperature'),
 'co2_solubility': task('co2solubility_S.xlsx', 'co2solubility', 'x_CO2', 'mole fraction',
                        'T (K)', 'P (bar)', pressure_factor=.1, pressure_unit='MPa'),
}
