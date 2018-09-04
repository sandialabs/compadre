import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import canga_driver

os.chdir(parentdir)
#canga_driver.execute_test("mpas_2562.nc","ne30np4_latlon.091226_mass_added.nc",porder=4,field_type=4,metric=0,use_obfet=1)
#canga_driver.execute_test("mpas_10242.nc","ne30np4_latlon.091226_mass_added.nc",porder=4,field_type=4,metric=0,use_obfet=1)
#canga_driver.execute_test("mpas_40962.nc","ne30np4_latlon.091226_mass_added.nc",porder=4,field_type=4,metric=0,use_obfet=1)

#canga_driver.execute_test("mpas_2562.nc","homme_ne8np4_mass_added.nc",porder=2,field_type=4,metric=0,use_obfet=0)
#canga_driver.execute_test("mpas_10242.nc","homme_ne16np4_mass_added.nc",porder=2,field_type=4,metric=0,use_obfet=0)
#canga_driver.execute_test("mpas_40962.nc","homme_ne32np4_mass_added.nc",porder=2,field_type=4,metric=0,use_obfet=0)
#canga_driver.execute_test("mpas_163842.nc","homme_ne64np4_mass_added.nc",porder=2,field_type=4,metric=0,use_obfet=0)


#canga_driver.execute_test("mpas_2562.nc","homme_ne8np4_mass_added.nc",porder=6,field_type=0,metric=2,use_obfet=1)
#canga_driver.execute_test("mpas_10242.nc","homme_ne16np4_mass_added.nc",porder=6,field_type=0,metric=2,use_obfet=1)
#canga_driver.execute_test("mpas_40962.nc","homme_ne32np4_mass_added.nc",porder=6,field_type=0,metric=2,use_obfet=1)
#canga_driver.execute_test("mpas_163842.nc","homme_ne64np4_mass_added.nc",porder=6,field_type=0,metric=2,use_obfet=1)
canga_driver.execute_test("homme_ne8np4_mass_added.nc","mpas_2562.nc",porder=6,field_type=0,metric=2,use_obfet=1)
canga_driver.execute_test("homme_ne16np4_mass_added.nc","mpas_10242.nc",porder=6,field_type=0,metric=2,use_obfet=1)
canga_driver.execute_test("homme_ne32np4_mass_added.nc","mpas_40962.nc",porder=6,field_type=0,metric=2,use_obfet=1)
canga_driver.execute_test("homme_ne64np4_mass_added.nc","mpas_163842.nc",porder=6,field_type=0,metric=2,use_obfet=1)
