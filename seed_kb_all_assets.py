import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
django.setup()

from django.utils import timezone
from core.models import (
    Tenant, Product, Asset, ProductDomain, ProductCategory, Brand,
    KnowledgeDocument, KnowledgeChunk, ServiceCall, ServiceResolutionIndex
)
from core.services.indexing import index_document, reindex_all_resolutions
from core.services.chroma_store import sync_document_to_chroma

tenant = Tenant.objects.get(name="Starlly Tester")
print(f"Targeting tenant: {tenant.name} (ID: {tenant.id})")

# Cache taxonomy
domains = {d.name: d for d in ProductDomain.objects.filter(tenant=tenant)}
categories = {c.name: c for c in ProductCategory.objects.filter(tenant=tenant)}
products = {p.name: p for p in Product.objects.filter(tenant=tenant)}

print("Loaded products:", list(products.keys()))

DOCUMENTS = [
    # -------------------------------------------------------------------------
    # AC SUPPLIER (Office Equipment / Power Supply Unit)
    # -------------------------------------------------------------------------
    {
        "title": "AC Supplier Troubleshooting and Power Diagnostic Guide",
        "description": "Comprehensive troubleshooting guide for AC Supplier power distribution units, covering power failure, overload tripping, voltage regulation, and inverter diagnostics.",
        "product_name": "AC Supplier",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "troubleshooting",
        "tags": "ac-supplier, power-supply, power-not-working, no-power, voltage, overload, breaker-tripped, inverter, fuse, troubleshooting",
        "content": """# AC Supplier Troubleshooting and Power Diagnostic Guide

## AC Supplier Not Working or No Output Voltage
Check the main 230V AC mains wall receptacle and master circuit breaker. Inspect the input power cord for physical damage, pinching, or scorched insulation. Verify if the front panel AC input indicator LED is illuminated green.
Step 1 - Switch off the master circuit breaker and disconnect the AC input plug before inspection.
Why: Prevents accidental electrical shock, short circuits, or arc flash hazards during internal fuse and terminal inspection.
Step 2 - Inspect the 15A primary slow-blow ceramic fuse located inside the rear AC power entry module drawer.
Why: Severe power line transients, lightning surges, and sudden grid restoration spikes blow this sacrificial fuse to isolate internal transformer windings.
Step 3 - Measure output terminal voltage across live and neutral using a calibrated True-RMS digital multimeter.
Why: Confirms whether the internal inverter regulator board is generating clean 230V AC within the acceptable +/- 5% operating window.
Step 4 - Verify the high-flow cooling fan is spinning freely and intake air filter grilles are unclogged.
Why: Internal bimetallic thermal cutoff switches trip automatically when the main heatsink temperature exceeds 75°C.
Expected result: Front panel AC output indicator illuminates steady green and stable 230V AC is restored across all distribution ports.
Create a field service call if the circuit breaker repeatedly trips immediately upon switching on or if an acrid burning odor is detected.

## Overload Tripping and Automatic Circuit Breaker Activation
Check total connected equipment wattage across all active distribution sockets against the rated 1500W continuous capacity. Inspect downstream load cables for pinched insulation or short circuits.
Step 1 - Disconnect all connected equipment plugs from the AC supplier distribution channels.
Why: Isolates whether the tripping fault originates inside the AC supplier or is caused by downstream field equipment.
Step 2 - Reset the thermal circuit breaker push-button on the rear panel by pressing firmly after allowing a 5-minute cool-down period.
Why: Bimetallic thermal trip mechanisms require internal temperature normalization before the mechanical latch can re-engage safely.
Step 3 - Reconnect downstream loads one at a time while monitoring current draw on the digital ammeter.
Why: Identifies the exact faulty branch circuit or overloading device causing excessive inrush current.
Expected result: The AC supplier maintains continuous operation without breaker disengagement under nominal connected loads.
Create a service call if the circuit breaker trips with zero loads connected to any socket.

## AC Voltage Fluctuations and Output Sagging
Verify building phase voltage and neutral-to-ground potential at the main distribution board.
Step 1 - Measure input supply voltage under load to ensure line potential does not drop below 195V AC.
Why: Excessive building line voltage drop starves the power conditioning transformer and induces harmonic distortion.
Step 2 - Inspect the automatic voltage regulation (AVR) relay module and tighten all terminal screw blocks to 1.8 Nm.
Why: Loose terminal screws create high-resistance contacts that cause voltage drop and localized resistive heating.
Step 3 - Clean dust accumulations off the transformer windings with dry compressed air under 30 PSI.
Why: Conductive dust particles cause eddy current leakage and degrade magnetic core efficiency.
Expected result: Output voltage stabilizes at 230V AC +/- 3% under dynamic load shifts.
Create a service call if output voltage remains below 205V AC despite normal 230V AC input supply.
"""
    },
    {
        "title": "AC Supplier Installation, Electrical Wiring & Operating Manual",
        "description": "Standard operating and installation procedure for AC Supplier units, covering earthing, circuit breaker ratings, and load distribution.",
        "product_name": "AC Supplier",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "installation",
        "tags": "ac-supplier, installation, wiring, commissioning, earthing, specifications",
        "content": """# AC Supplier Installation, Electrical Wiring & Operating Manual

## Site Preparation and Earthing
1. Confirm the installation location is clean, dry, and provides at least 20 cm clearance on all sides for convective airflow.
2. Confirm the main AC supply provides dedicated 230V single-phase power with solid earth ground resistance under 2.0 ohms.
3. Verify the asset model number, serial number, and voltage rating match the facility engineering plan.

## Installation Steps
1. Mount the AC supplier securely in the standard 19-inch equipment rack or flat utility shelf using supplied brackets.
2. Connect the dedicated copper earth ground bonding cable directly from the chassis grounding stud to the facility earth bar.
3. Connect the primary AC input cable to the protected wall isolator switch.
4. Distribute downstream equipment cables across Rail A and Rail B to balance internal transformer loading evenly.
5. Turn on the master switch and confirm the output voltage reads 230V +/- 5% on the front panel display.

## Commissioning Check
Record the asset code, installation date, baseline input voltage, and connected load wattage. Confirm the unit runs continuously for 30 minutes with heatsink temperature remaining below 45°C.
Create a service call if the unit buzzes loudly, fails grounding verification, or output voltage fluctuates erratically.
"""
    },
    {
        "title": "AC Supplier Preventive Maintenance & Inspection SOP",
        "description": "Routine preventive maintenance schedule for AC Supplier power distribution units.",
        "product_name": "AC Supplier",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "maintenance",
        "tags": "ac-supplier, maintenance, inspection, cleaning, periodic-service",
        "content": """# AC Supplier Preventive Maintenance & Inspection SOP

## Every Month
Inspect cooling fan rotation, vacuum intake filter screens, and check power cable insulation for thermal discoloration.

## Every 3 Months
Check terminal block tightness with an insulated torque screwdriver (1.8 Nm). Record input voltage, output voltage, and load current under peak operational conditions.

## Every Year
A certified technician must perform an insulation resistance (Megger) test between live conductors and chassis ground (minimum 10 Megohms at 500V DC). Clean internal heatsinks and inspect electrolytic filter capacitors for electrolyte leakage or bulging tops.
"""
    },

    # -------------------------------------------------------------------------
    # MILK CHILLER (Dairy Equipment / Milk Cooling)
    # -------------------------------------------------------------------------
    {
        "title": "Milk Chiller Troubleshooting and Refrigeration Diagnostic Guide",
        "description": "Detailed troubleshooting guide for industrial bulk milk chillers and cooling tanks, covering compressor failure, temperature rise, agitator issues, and refrigerant leaks.",
        "product_name": "Milk Chiller",
        "category_name": "Milk Cooling",
        "domain_name": "Dairy Equipment",
        "doc_type": "troubleshooting",
        "tags": "milk-chiller, chiller, cooling-issue, temperature-rising, compressor, agitator, refrigeration, refrigerant, E-02, troubleshooting",
        "content": """# Milk Chiller Troubleshooting and Refrigeration Diagnostic Guide

## Milk Chiller Not Cooling or Temperature Rising Above 4°C
Check the digital temperature controller display for active alarm codes. Inspect the main 3-phase power disconnect switch and confirm all 3 phases have normal voltage.
Step 1 - Verify that the outdoor condensing unit exhaust fans and scroll compressor are actively humming.
Why: Failure of the outdoor condensing unit prevents heat transfer from the milk cooling plate to ambient air.
Step 2 - Clean the condenser coil aluminum fins using a soft nylon brush or low-pressure air to clear dust and farm debris.
Why: Clogged condenser fins cause high head pressure that triggers the high-pressure safety cutout switch.
Step 3 - Inspect the refrigerant sight glass for continuous clear liquid flow without vigorous bubbles.
Why: Persistent foaming bubbles in the sight glass indicate a refrigerant leak or low R404A charge.
Step 4 - Inspect the milk tank agitator paddle to confirm it is rotating continuously at 28 RPM.
Why: Uniform milk agitation is mandatory to break the thermal boundary layer and prevent milk freezing on the evaporator plate.
Expected result: Milk bulk temperature drops steadily toward the 4.0°C holding setpoint at a minimum cooling rate of 1°C per 15 minutes.
Create a field service call if the compressor clicks repeatedly without starting or if oily residue is discovered around copper piping joints.

## Agitator Motor Not Running or Scraping Tank Wall
Check agitator safety interlock microswitch on the tank lid.
Step 1 - Ensure the tank inspection hatch is fully latched down.
Why: An open hatch trips the operator safety cutoff switch and disables agitator paddle rotation.
Step 2 - Inspect the gearbox oil level sight glass and listen for mechanical gear grinding.
Why: Insufficient gear lubrication causes worm gear seizure and motor thermal overload.
Step 3 - Manually rotate the agitator shaft by hand while power is isolated to check paddle clearance.
Why: Detects bent drive shafts or mechanical interference with the tank bottom dimple jacket.
Expected result: Agitator rotates smoothly and quietly without shaft wobble or wall contact.
Create a service call if the motor hums without rotating or if metal shavings are visible in the gearbox drip tray.

## Temperature Controller Error E-04 Sensor Fault
Check probe wire routing from the tank thermowell to the control box.
Step 1 - Inspect the PT100 temperature sensor cable for pinching, moisture ingress, or terminal corrosion.
Why: Resistance changes caused by moisture or loose terminals simulate out-of-range temperature readings.
Step 2 - Verify sensor resistance with a digital ohmmeter (PT100 should measure 100 ohms at 0°C and approx 101.6 ohms at 4°C).
Why: Confirms sensor calibration accuracy and detects open-circuit platinum RTD elements.
Expected result: Digital controller reads accurate milk temperature within +/- 0.2°C.
Create a service call if the sensor measures infinite resistance or if error E-04 persists after terminal tightening.
"""
    },
    {
        "title": "Milk Chiller Operating Manual & Daily CIP Sanitation Guide",
        "description": "Standard operating procedures, cleaning-in-place (CIP) sanitation, and temperature recording protocols for milk chillers.",
        "product_name": "Milk Chiller",
        "category_name": "Milk Cooling",
        "domain_name": "Dairy Equipment",
        "doc_type": "user_guide",
        "tags": "milk-chiller, CIP, cleaning, sanitation, operating, daily-procedure",
        "content": """# Milk Chiller Operating Manual & Daily CIP Sanitation Guide

## Daily Startup & Milk Reception
1. Verify the chiller interior is sanitized and dry before milk collection starts.
2. Ensure the bottom drain butterfly valve is fully closed and clamped.
3. Switch the cooling mode to AUTO once milk level covers the agitator paddle and evaporator plate.
4. Never activate refrigeration on an empty tank as ice will form on the evaporator surface.

## Cleaning-in-Place (CIP) Procedure
1. Empty all collected milk and perform an initial warm-water (38°C-40°C) pre-rinse until drainage runs clear.
2. Circulate approved alkaline detergent solution (1.5% concentration at 65°C-70°C) through the rotating spray ball for 15 minutes.
3. Drain detergent and rinse thoroughly with cold potable water.
4. Circulate approved sanitizing acid solution (0.5% concentration at ambient temperature) for 5 minutes.
5. Perform final potable water rinse and inspect the tank interior with a clean inspection flashlight.
"""
    },
    {
        "title": "Milk Chiller Preventive Maintenance & Refrigeration Servicing Guide",
        "description": "Technical preventive maintenance guide for refrigeration compressors, expansion valves, and thermal insulation.",
        "product_name": "Milk Chiller",
        "category_name": "Milk Cooling",
        "domain_name": "Dairy Equipment",
        "doc_type": "maintenance",
        "tags": "milk-chiller, maintenance, compressor, refrigeration, servicing",
        "content": """# Milk Chiller Preventive Maintenance & Refrigeration Servicing Guide

## Monthly Maintenance
Inspect agitator shaft mechanical seal for milk leakage. Check compressor oil sight glass (oil level must be between 1/3 and 2/3 of sight glass). Clean air-cooled condenser coil with fin comb and vacuum.

## Quarterly Maintenance
Check thermostatic expansion valve (TXV) superheat setting (target 5K to 8K). Perform electronic halogen leak detection on all brazed refrigeration joints. Measure compressor running amps on all three phases.

## Annual Maintenance
Drain and replace agitator gearbox synthetic food-grade lubricant (ISO VG 220). Inspect electrical contactor contacts for pitting and replace if worn. Recalibrate digital temperature sensor using an ice bath reference (0.0°C).
"""
    },

    # -------------------------------------------------------------------------
    # PALLET LIFT (Material Handling / Pallet Equipment)
    # -------------------------------------------------------------------------
    {
        "title": "Pallet Lift Troubleshooting & Hydraulic Diagnostic Guide",
        "description": "Field troubleshooting guide for Pallet Lift equipment, covering lift cylinder failure, hydraulic pressure drop, slow lifting, and drive wheel errors.",
        "product_name": "Pallet Lift",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "troubleshooting",
        "tags": "pallet-lift, hydraulic, lifting, forks-not-raising, pressure-loss, slow-lift, battery, troubleshooting",
        "content": """# Pallet Lift Troubleshooting & Hydraulic Diagnostic Guide

## Pallet Lift Will Not Raise or Drops Slowly Under Load
Check the battery charge state indicator on the LED meter. Inspect the floor for hydraulic oil puddles under the chassis.
Step 1 - Check hydraulic fluid level in the transparent reservoir tank with forks completely lowered to the floor.
Why: Low hydraulic fluid causes pump cavitation and aerates the oil, preventing lifting pressure generation.
Step 2 - Inspect the hydraulic lift cylinder rod for scratches, pitting, or oil weeping around the polyurethane seal.
Why: Damaged rod seals allow pressurized oil to bypass the piston and cause uncontrolled fork drift.
Step 3 - Test the manual emergency lowering valve to verify it is fully closed and not leaking internally.
Why: A valve seated with grit or a loose manual knob permits hydraulic return flow straight to the tank.
Step 4 - Clean the suction strainer inside the hydraulic power pack reservoir.
Why: Clogged intake strainers restrict fluid flow to the gear pump and cause screaming pump cavitation noise.
Expected result: Forks raise smoothly to maximum lift height (200 mm) under full 2000 kg rated payload without drift.
Create a service call if the lift motor runs but forks fail to elevate, or if hydraulic hoses exhibit bulges.

## Traction Drive Inoperative or Steering Error E-12
Check the red emergency stop mushroom button on the tiller head.
Step 1 - Pull out the emergency stop button and turn the ignition key switch clockwise.
Why: A depressed E-stop switch mechanically disconnects the main power contactor coil.
Step 2 - Inspect the belly-button reverse safety switch on the tiller control handle.
Why: A stuck reverse safety microswitch locks out forward travel commands to protect the operator.
Step 3 - Check battery terminal lugs for white lead-sulfate corrosion or loose bolt connections.
Why: High contact resistance causes voltage drop under initial traction motor inrush current.
Expected result: Traction drive engages smoothly in forward and reverse speeds according to tiller throttle angle.
Create a service call if the drive wheel locks mechanically or motor brushes emit excessive sparking.
"""
    },
    {
        "title": "Pallet Lift Operator Safety & Daily Inspection Manual",
        "description": "Daily safety checklist, operating limits, and battery charging guidelines for pallet lifts.",
        "product_name": "Pallet Lift",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "user_guide",
        "tags": "pallet-lift, safety, operator-manual, checklist, daily-inspection",
        "content": """# Pallet Lift Operator Safety & Daily Inspection Manual

## Pre-Shift Inspection Checklist
1. Inspect fork tips for cracks, bending, or uneven height.
2. Verify smooth function of the dead-man braking handle when released.
3. Test the emergency belly-button reverse switch by walking backward into a soft obstacle.
4. Confirm horn sounds clearly and battery state of charge is above 30%.
5. Never exceed the rated load capacity of 2000 kg or transport passengers on forks.
"""
    },
    {
        "title": "Pallet Lift Hydraulic Fluid & Drive Maintenance SOP",
        "description": "Preventive maintenance procedures for hydraulic pumps, cylinders, chains, and battery cells.",
        "product_name": "Pallet Lift",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "maintenance",
        "tags": "pallet-lift, maintenance, hydraulic-fluid, battery, drive-wheel",
        "content": """# Pallet Lift Hydraulic Fluid & Drive Maintenance SOP

## Monthly Maintenance
Clean battery terminals and top up distilled water in flooded lead-acid cells. Inspect lift linkages, grease wheel bearings with NLGI 2 lithium grease, and check tire tread wear.

## Annual Maintenance
Drain and replace hydraulic fluid with clean ISO VG 32 hydraulic oil. Replace high-pressure hydraulic return filter. Inspect motor carbon brushes and commutator surface. Retorque chassis structural bolts.
"""
    },

    # -------------------------------------------------------------------------
    # AUTOPICK (Material Handling / Pallet Equipment)
    # -------------------------------------------------------------------------
    {
        "title": "Autopick Robotic Arm Troubleshooting & Error Code Guide",
        "description": "Technical troubleshooting guide for Autopick robotic palletizing and picking systems, covering axis jams, gripper failures, sensor alignment, and PLC timeouts.",
        "product_name": "Autopick",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "troubleshooting",
        "tags": "autopick, robotic-arm, pick-and-place, E-402, axis-jam, gripper, vacuum, sensor, troubleshooting",
        "content": """# Autopick Robotic Arm Troubleshooting & Error Code Guide

## Autopick Error E-402 Axis Jam or Joint Overcurrent
Check the working envelope for mechanical obstacles or pallet misalignment. Inspect robot base emergency stop circuit.
Step 1 - Disengage the automated run mode and switch the teach pendant controller to manual slow-speed jog mode.
Why: Manual jog allows safe step-by-step axis movement at 10% velocity to isolate the binding joint.
Step 2 - Inspect the harmonic drive and timing belts on Joint 2 and Joint 3 for debris or broken teeth.
Why: Foreign object ingress into gear tooth profiles locks the servo motor and triggers axis torque overcurrent limits.
Step 3 - Verify optical limit switches and magnetic home position markers are clean and aligned.
Why: Contaminated optical sensors prevent the controller from detecting axis reference zero during homing routines.
Expected result: All 6 axes execute homing calibration smoothly without servo amplifier overtemperature or torque alarms.
Create a field service call if a joint makes screeching bearing noises or if the servo drive displays encoder feedback loss error E-408.

## Gripper Fails to Pick or Drops Items During Transfer
Check pneumatic line pressure gauge on the robot tool flange regulator (must read 6.0 to 6.5 bar).
Step 1 - Inspect suction cups or mechanical gripper finger pads for tears, oil glazing, or wear.
Why: Degraded elastomer seals prevent vacuum seal formation on cardboard or plastic tote surfaces.
Step 2 - Test the pneumatic solenoid valve actuating signals using the teach pendant I/O monitor.
Why: Confirms whether the PLC digital output card is transmitting the 24V DC pick command to the tool head.
Step 3 - Clean the vacuum venturi generator nozzle and exhaust silencer filter.
Why: Dust accumulation in the venturi throat reduces vacuum level below the required -80 kPa threshold.
Expected result: Gripper generates instantaneous vacuum clamp and holds target payload securely through maximum-acceleration trajectory.
Create a service call if the vacuum sensor reports low pressure despite clean nozzles or if gripper fingers are bent.
"""
    },
    {
        "title": "Autopick Sensor Calibration & Teaching Manual",
        "description": "Procedures for setting up picking coordinates, teaching waypoints, and aligning safety light curtains.",
        "product_name": "Autopick",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "installation",
        "tags": "autopick, calibration, teaching, coordinates, light-curtain",
        "content": """# Autopick Sensor Calibration & Teaching Manual

## Safety Perimeter & Light Curtain Alignment
1. Verify safety perimeter light curtains are mounted securely and aligned with transmitter/receiver optical axes.
2. Confirm breaking the light curtain beam immediately halts robot motion within 120 ms.
3. Establish safety interlock zones before teaching waypoints.

## Coordinate Teaching Procedure
1. Enter teach mode using the safety enabling switch (3-position deadman switch on teach pendant).
2. Jog tool center point (TCP) to pallet datum corner pin and record Base Point P0.
3. Define picking matrix grid dimensions and pallet stacking layers.
4. Perform slow-speed test cycle with zero payload before commissioning full production speed.
"""
    },
    {
        "title": "Autopick Preventive Maintenance & Joint Lubrication Schedule",
        "description": "Routine lubrication, cable harness inspection, and backlash checking for Autopick robotic arms.",
        "product_name": "Autopick",
        "category_name": "Pallet Equipment",
        "domain_name": "Material Handling",
        "doc_type": "maintenance",
        "tags": "autopick, maintenance, lubrication, gear-grease, inspection",
        "content": """# Autopick Preventive Maintenance & Joint Lubrication Schedule

## Every 500 Operating Hours
Inspect pneumatic hoses on the dress pack for friction rub marks. Clean optical sensors and reflectors. Drain water from pneumatic filter regulator bowl.

## Every 2000 Operating Hours
Replenish RV reduction gear grease with Kyodo Yushi Molywhite RE00 grease using dedicated purge ports. Measure mechanical backlash on Axes 1 through 4 (maximum allowable backlash 0.05 mm). Check internal cable harness flexing loops.
"""
    },

    # -------------------------------------------------------------------------
    # VOLTCORE (Office Equipment / UPS & Voltage Stabilizer)
    # -------------------------------------------------------------------------
    {
        "title": "Voltcore UPS & Voltage Stabilizer Troubleshooting Guide",
        "description": "Field diagnostics for Voltcore industrial UPS systems, covering inverter faults, battery charging failure, bypass mode transfers, and alarm codes.",
        "product_name": "Voltcore",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "troubleshooting",
        "tags": "voltcore, ups, inverter, battery-not-charging, F-07, bypass, voltage-stabilizer, alarm, beeping, troubleshooting",
        "content": """# Voltcore UPS & Voltage Stabilizer Troubleshooting Guide

## Voltcore Inverter Fault F-07 or Beeping Audible Alarm
Check the LCD screen for active fault codes and battery string voltage. Verify mains input breaker is closed.
Step 1 - Observe the alarm pattern: continuous tone indicates critical inverter trip; pulsed tone indicates utility power failure.
Why: Decodes the internal controller state to determine if operation is on inverter, static bypass, or battery discharge.
Step 2 - Inspect battery bank DC circuit breaker and interconnect busbar cables for discoloration or heating.
Why: Loose battery terminals cause high voltage drop and false battery low voltage cutoff warnings.
Step 3 - Check the cooling fan status on the inverter power module heat sinks.
Why: Blocked rear exhaust vents trigger thermal protection sensors and force the unit into emergency bypass mode.
Step 4 - Verify load percentage on the front panel display does not exceed 100% capacity rating.
Why: Sustained overload above 110% will transfer the load to raw bypass after 60 seconds to protect inverter IGBT transistors.
Expected result: Inverter status LED glows steady green, output delivers pure sinusoidal 230V AC, and alarm ceases.
Create a field service call if fault F-07 persists after clearing load or if battery cells show bulging cases.

## Battery Bank Fails to Hold Load During Power Outage
Verify individual cell float voltages using a digital multimeter across all series battery terminals.
Step 1 - Perform an offline battery discharge impedance test across each 12V VRLA battery module.
Why: Identifies high-impedance failed cells that drag down the entire series battery string under load.
Step 2 - Inspect the DC float charging voltage from the internal rectifier (should maintain 54.4V DC for a 48V string).
Why: An undercharging rectifier leaves batteries in a partially discharged state prone to sulfation.
Step 3 - Clean battery posts with baking soda solution and retorque terminal bolts to 4.5 Nm.
Why: Lead sulfate buildup insulates the cable lugs and prevents high current delivery during mains loss.
Expected result: Battery string supports full critical load for the specified 30-minute runtime specification.
Create a service call if battery string voltage drops below 42V DC within 2 minutes of mains disconnect.
"""
    },
    {
        "title": "Voltcore Installation & Power Distribution Wiring Guide",
        "description": "Wiring instructions, neutral bonding, circuit breaker sizing, and battery cabinet connections for Voltcore units.",
        "product_name": "Voltcore",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "installation",
        "tags": "voltcore, installation, wiring, neutral-bonding, battery-cabinet",
        "content": """# Voltcore Installation & Power Distribution Wiring Guide

## Pre-Installation Electrical Requirements
1. Install a dedicated 32A Type D curve input circuit breaker at the facility distribution panel.
2. Verify neutral-to-earth voltage does not exceed 1.5V AC under full load.
3. Install external battery cabinets within 2 meters of the main UPS chassis using 16 mm2 copper cables.
4. Ensure room ambient temperature is controlled between 20°C and 25°C to maximize battery lifespan.
"""
    },
    {
        "title": "Voltcore Battery Bank Maintenance & Impedance Testing SOP",
        "description": "Preventive maintenance protocols for VRLA battery strings, thermal imaging of busbars, and capacity discharge testing.",
        "product_name": "Voltcore",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "maintenance",
        "tags": "voltcore, maintenance, battery-bank, impedance-testing, thermal-imaging",
        "content": """# Voltcore Battery Bank Maintenance & Impedance Testing SOP

## Quarterly Inspection
Measure and record individual cell float voltages (normal 13.5V to 13.8V per 12V block). Inspect terminals for corrosion, leaks, or swelling. Perform infrared thermal scan of battery links under float charge.

## Annual Load Bank Test
Conduct a controlled 80% depth-of-discharge test with a dummy resistive load bank. Calculate remaining battery capacity against nameplate Ah rating. Replace battery strings exceeding 3 years of service or showing below 80% capacity retention.
"""
    },

    # -------------------------------------------------------------------------
    # PAPER SHREDDER (Office Equipment / Elephanta & CMC1000)
    # -------------------------------------------------------------------------
    {
        "title": "Paper Shredder & CMC1000 Troubleshooting Guide",
        "description": "Diagnostics and repair procedures for commercial paper shredders, covering paper jams, motor stalling, auto-reverse loops, and sensor cleaning.",
        "product_name": "Paper Shredder",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "troubleshooting",
        "tags": "paper-shredder, cmc1000, paper-jam, motor-humming, auto-reverse, cutter-jammed, shredder, troubleshooting",
        "content": """# Paper Shredder & CMC1000 Troubleshooting Guide

## Paper Shredder Jammed or Motor Humming Without Rotation
Check the auto-feed throat for over-capacity paper bundles. Turn power switch to OFF immediately.
Step 1 - Slide the mode switch from AUTO to REVERSE mode to back out trapped paper fibers.
Why: Reverse rotation unrolls compressed sheets from between the interlocking cross-cut steel cylinder teeth.
Step 2 - Disconnect AC power plug and carefully extract torn paper remnants using non-metallic tweezers.
Why: Disconnecting mains power guarantees personal safety and prevents sudden blade movement.
Step 3 - Apply 10 ml of approved vegetable-based shredder lubricating oil across the entire length of the feed throat.
Why: Lubricant reduces inter-blade metal friction and dissolves compacted paper dust binding the cutter knives.
Step 4 - Switch back to AUTO mode and feed a single clean sheet of paper to distribute the lubricant.
Why: Verifies that full cutting torque has been restored without straining the motor drive gears.
Expected result: Cutter head rotates freely, shredding paper into compliant 4x40 mm cross-cut particles without stalling.
Create a service call if the motor hums in both forward and reverse without movement or if gears sound stripped.

## Shredder Will Not Turn On or Thermal Indicator Illuminated
Check waste bin position and bin-full optical sensor window.
Step 1 - Pull out the waste collection bin, empty all shredded material, and slide it firmly back into the chassis.
Why: An open bin interlock microswitch completely interrupts line power to the control PCB.
Step 2 - Wipe the clear optical bin-level sensors located inside the cutter cavity using a dry cotton swab.
Why: Fine paper dust coatings on infrared emitter lenses mimic a full waste bin and prevent motor start.
Step 3 - If the yellow thermal icon is lit, allow the unit to cool down for 25 minutes before restarting.
Why: Heavy duty shredders have motor thermal cutouts that require 20-30 minutes to auto-reset.
Expected result: Power indicator glows solid green and feeding paper activates automatic photo-sensor start.
Create a service call if the motor remains unresponsive after a 1-hour cool-down period.
"""
    },
    {
        "title": "Paper Shredder Safe Operation & Daily Maintenance Guide",
        "description": "Operating safety instructions, sheet capacity ratings, and daily blade oiling guidelines.",
        "product_name": "Paper Shredder",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "user_guide",
        "tags": "paper-shredder, operation, maintenance, oiling, safety",
        "content": """# Paper Shredder Safe Operation & Daily Maintenance Guide

## Operating Rules
1. Never exceed the rated 22-sheet capacity per pass (70 gsm paper).
2. Remove heavy industrial binder clips, cardboard, and vinyl sheets before shredding.
3. Keep ties, necklaces, loose clothing, and long hair strictly clear of the intake throat.
4. Empty waste receptacle when it reaches 75% full to prevent shredded paper backing up into the cutting knives.
5. Apply approved shredder oil every 30 minutes of continuous shredding.
"""
    },

    # -------------------------------------------------------------------------
    # EXTERIOR JACK (Material Handling / Lift Systems / ABB)
    # -------------------------------------------------------------------------
    {
        "title": "Exterior Jack Troubleshooting & Mechanical Diagnostic Guide",
        "description": "Troubleshooting guide for Exterior Jack equipment, covering motor failure, limit switch errors, hydraulic check valves, and manual overrides.",
        "product_name": "Exterior Jack",
        "category_name": "Lift Systems",
        "domain_name": "Material Handling",
        "doc_type": "troubleshooting",
        "tags": "exterior-jack, jack, lifting, limit-switch, motor-not-extending, drift, dcmotor, troubleshooting",
        "content": """# Exterior Jack Troubleshooting & Mechanical Diagnostic Guide

## Exterior Jack Will Not Extend or Retract
Check the 24V DC auxiliary power supply cable and master control pendant connection.
Step 1 - Inspect the pendant handheld controller cable for crimping, severed conductors, or moisture ingress.
Why: Moisture inside the push-button pendant causes signal shorting and disables up/down solenoid contactors.
Step 2 - Verify that the mechanical limit switches at full-retract and full-extend positions are not physically stuck.
Why: A mechanically frozen limit switch sends an active limit signal that blocks motor actuation in that direction.
Step 3 - Measure battery voltage at the 24V motor input terminals while pressing the extend button.
Why: Voltage dropping below 21V under load indicates high internal cable resistance or weak DC power supply.
Step 4 - If power is confirmed but motor will not turn, use the manual hex override socket with a socket wrench.
Why: Frees mechanical binding in the ball-screw drive and confirms whether the gearbox is locked.
Expected result: Jack foot extends smoothly at rated 15 mm/sec speed until grounding on stabilization pad.
Create a service call if the motor exhibits severe vibration, smoke, or stripped internal acme screw threads.

## Jack Drifts or Loses Height Under Static Load
Check for mechanical lock pin engagement and hydraulic check valve sealing.
Step 1 - Ensure the secondary mechanical locking collar or safety pin is fully inserted into the mast hole.
Why: Mechanical locks provide fail-safe load retention independent of hydraulic or electrical power.
Step 2 - Inspect the load-holding check valve for internal fluid bypass or seat contamination.
Why: Contaminants on the check valve poppet prevent complete closure and permit slow backflow.
Expected result: Jack maintains zero height loss over a 24-hour test period under maximum rated 10-ton working load.
Create a service call if fluid leaks past the main gland nut or locking collar fails to latch.
"""
    },
    {
        "title": "Exterior Jack Installation, Handover & Load Rating Guide",
        "description": "Installation requirements, ground pad load calculations, and commissioning procedure for Exterior Jacks.",
        "product_name": "Exterior Jack",
        "category_name": "Lift Systems",
        "domain_name": "Material Handling",
        "doc_type": "installation",
        "tags": "exterior-jack, installation, commissioning, load-rating, outrigger",
        "content": """# Exterior Jack Installation, Handover & Load Rating Guide

## Structural Mounting Requirements
1. Fasten jack mounting plates to reinforced structural chassis members using Grade 8.8 high-tensile M16 bolts.
2. Confirm outrigger landing pads have a minimum footprint of 300x300 mm to distribute ground contact pressure.
3. Verify electrical supply wires are protected with flexible UV-resistant conduit.
4. Perform 125% overload proof test (12.5 tons static) before customer handover.
"""
    },

    # -------------------------------------------------------------------------
    # ASUS CELERON (Office Equipment / Diagnostics Workstation)
    # -------------------------------------------------------------------------
    {
        "title": "Asus Celeron & Service Terminal Troubleshooting Guide",
        "description": "Hardware and diagnostic port troubleshooting guide for Asus Celeron workstations and Samsung Service Terminals.",
        "product_name": "Asus Celeron",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "troubleshooting",
        "tags": "asus-celeron, terminal, workstation, black-screen, shutdown, com-port, serial, usb, troubleshooting",
        "content": """# Asus Celeron & Service Terminal Troubleshooting Guide

## Terminal Screen Black or System Unexpectedly Shuts Down
Check the external 19V DC power brick indicator LED and mains socket connection.
Step 1 - Verify the DC barrel connector is firmly seated into the chassis power jack without play.
Why: A loose barrel connection causes momentary power disconnects and abrupt system resets.
Step 2 - Inspect rear exhaust vents for heavy dust blankets and listen for the internal cooling fan.
Why: Thermal protection circuitry initiates automatic CPU shutdown when core temperature exceeds 85°C.
Step 3 - Perform a hard hardware power reset: disconnect power, hold the power button down for 30 seconds, then reconnect.
Why: Discharges residual capacitance on motherboard power management ICs and clears latched fault states.
Step 4 - Connect an external HDMI monitor to verify if the fault is display panel failure or system POST failure.
Why: Isolates whether the motherboard and OS are running normally despite an LCD backlight failure.
Expected result: System boots past BIOS POST screen to the Servy diagnostics desktop within 45 seconds.
Create a service call if the motherboard power LED blinks a red hardware diagnostic code or display shows GPU artifacts.

## Diagnostic Port / USB Field Equipment Communication Lost
Check the USB-to-RS485 interface converter cable connecting to field instruments.
Step 1 - Open Device Manager and verify the COM port entry does not display a yellow exclamation mark.
Why: Confirms that the FTDI serial bus driver is loaded and operating without IRQ conflicts.
Step 2 - Unplug and reconnect the diagnostic interface cable into a different USB 3.0 port on the chassis.
Why: Forces USB controller re-enumeration and resets the serial bus transceiver buffer.
Step 3 - Verify communication baud rate (9600 bps, 8 data bits, no parity, 1 stop bit) matches the field asset spec.
Why: Mismatched serial framing parameters cause buffer framing errors and communication timeout errors.
Expected result: Diagnostic software establishes real-time telemetry link with zero packet loss.
Create a service call if USB controller ports fail to deliver 5V bus power or serial port drops packets continuously.
"""
    },
    {
        "title": "Asus Celeron Field Terminal Setup & Maintenance Guide",
        "description": "Operating system configuration, driver updates, and preventive dust maintenance for field terminals.",
        "product_name": "Asus Celeron",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "maintenance",
        "tags": "asus-celeron, setup, drivers, maintenance, clean-terminal",
        "content": """# Asus Celeron Field Terminal Setup & Maintenance Guide

## Monthly Preventive Maintenance
Clean cooling fan exhaust louvers using anti-static compressed air. Wipe touch screen digitizer with 70% isopropyl alcohol wipes. Verify backup of offline diagnostic logs to cloud server.
"""
    },

    # -------------------------------------------------------------------------
    # KTM (Office Equipment / Utility Vehicle & Mobile Power)
    # -------------------------------------------------------------------------
    {
        "title": "KTM Utility Vehicle & Engine Troubleshooting Guide",
        "description": "Troubleshooting manual for KTM utility transport and mobile power generation units, covering engine crank issues, battery drain, and drive chain.",
        "product_name": "KTM",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "troubleshooting",
        "tags": "ktm, utility-vehicle, engine-not-starting, battery-drain, overheating, chain-slack, fuel-system, troubleshooting",
        "content": """# KTM Utility Vehicle & Engine Troubleshooting Guide

## Engine Cranks But Fails to Start
Check fuel level in tank and verify the handlebar engine kill switch is in the RUN position.
Step 1 - Verify the 12V starter battery terminal voltage reads at least 12.6V DC in resting state.
Why: Insufficient cranking voltage prevents the electronic fuel injection (EFI) ECU from energizing ignition coils.
Step 2 - Listen for the high-pressure fuel pump prime whine for 3 seconds after turning the ignition key ON.
Why: An inoperative fuel pump indicates a blown 10A EFI fuse, failed fuel pump relay, or fuel pump motor failure.
Step 3 - Inspect the spark plug cap and HT lead for firm seating on the spark plug terminal.
Why: Loose spark plug caps cause spark energy dissipation and prevent fuel-air mixture combustion.
Step 4 - Inspect the air intake box filter element for heavy dust clogging or oil contamination.
Why: Severe intake air restriction starves the engine of required combustion airflow and fouls the plug.
Expected result: Engine starts cleanly within 2 seconds of starter button engagement and idles smoothly at 1400 RPM.
Create a service call if starter motor fails to spin the flywheel or if engine backfires through the throttle body.

## Engine Overheating and High Coolant Temperature Warning
Check coolant level in the translucent overflow reservoir tank when engine is cool.
Step 1 - Inspect radiator fins for mud, chaff, or insect buildup and wash carefully with low-pressure water.
Why: Blocked radiator airflow prevents heat rejection and causes rapid coolant boiling.
Step 2 - Verify that the electric radiator cooling fan spins up when coolant temperature reaches 95°C.
Why: Failed fan relay or thermal switch allows engine temperature to rise dangerously during low-speed plant idling.
Expected result: Temperature gauge stabilizes at normal operating temperature (85°C-90°C) during continuous plant patrol operations.
Create a service call if white exhaust smoke is observed or coolant leaks from the water pump weep hole.
"""
    },
    {
        "title": "KTM Pre-Shift Inspection & Periodic Servicing Schedule",
        "description": "Daily inspection procedure, tire pressures, drive chain tensioning, and oil change schedule for KTM utility vehicles.",
        "product_name": "KTM",
        "category_name": "Office Equipment",
        "domain_name": "Office & Utility Equipment",
        "doc_type": "maintenance",
        "tags": "ktm, maintenance, servicing, inspection, oil-change, chain",
        "content": """# KTM Pre-Shift Inspection & Periodic Servicing Schedule

## Daily Pre-Shift Checklist
1. Check tire inflation pressure (Front: 28 PSI, Rear: 32 PSI).
2. Check engine oil sight glass level on level ground.
3. Test front and rear hydraulic disc brakes for firm lever feel.
4. Verify drive chain free play is between 25 mm and 30 mm.
5. Inspect headlight, tail light, and turn indicators.
"""
    },
]

# Seed / update documents
created_count = 0
updated_count = 0

for doc_def in DOCUMENTS:
    prod = products.get(doc_def["product_name"])
    cat = categories.get(doc_def["category_name"])
    dom = domains.get(doc_def["domain_name"])

    doc, created = KnowledgeDocument.objects.update_or_create(
        tenant=tenant,
        title=doc_def["title"],
        defaults={
            "description": doc_def["description"],
            "domain": dom,
            "category": cat,
            "product": prod,
            "asset": None,  # product-level doc so all assets of this product inherit it
            "doc_type": doc_def["doc_type"],
            "source_type": "text",
            "tags": doc_def["tags"],
            "content_text": doc_def["content"],
            "is_confidential": False,
            "disable_sharing": False,
            "is_rag_enabled": True,
        }
    )
    if created:
        created_count += 1
    else:
        updated_count += 1

    chunks_count = index_document(doc)
    sync_document_to_chroma(doc.id)
    print(f"[{'CREATED' if created else 'UPDATED'}] Doc #{doc.id}: '{doc.title}' ({chunks_count} chunks indexed)")

# Also enhance existing Milk Analyzer MA-100 Troubleshooting Guide with AC Power section
milk_trouble = KnowledgeDocument.objects.filter(tenant=tenant, title="Milk Analyzer MA-100 Troubleshooting Guide").first()
if milk_trouble:
    ac_section = """
## AC Power Supply and Adapter Not Working
Check the external AC-DC power adapter brick and the wall outlet. Verify whether the green LED indicator on the AC adapter is illuminated.
Step 1 - Inspect the AC power cord and confirm the wall outlet is providing 230V AC mains power.
Why: Tripped circuit breakers or loose wall sockets prevent the power supply adapter from receiving primary voltage.
Step 2 - Verify that the DC output connector is securely screwed into the milk analyzer rear power jack.
Why: A loose or oxidized DC barrel connector causes intermittent power drops and resets the measurement processor.
Step 3 - Check the 2A slow-blow DC fuse located in the drawer adjacent to the power inlet.
Why: Line power surges or internal heater short circuits blow this fuse to safeguard the analytical board.
Step 4 - If connected to an external AC Supplier unit, test the outlet with a known-good lamp or meter.
Why: Isolates whether the fault is in the plant's auxiliary AC Supplier unit or the analyzer's internal power supply.
Expected result: The analyzer power indicator illuminates steady green and the startup diagnostic routine finishes.
Create a field service call if the adapter smells burnt, displays a blinking indicator, or powers off when the sample pump starts.
"""
    if "AC Power Supply and Adapter Not Working" not in milk_trouble.content_text:
        milk_trouble.content_text += ac_section
        milk_trouble.tags += ", ac-supplier, ac-power, adapter, power-supply"
        milk_trouble.save()
        c_count = index_document(milk_trouble)
        sync_document_to_chroma(milk_trouble.id)
        print(f"[ENHANCED] Milk Analyzer MA-100 Troubleshooting Guide (+AC power section, {c_count} chunks)")

# Also add realistic closed service calls for AC Supplier and other newly covered products
print("\nSeeding realistic closed ServiceCalls with fleet resolutions...")
now = timezone.now()

SAMPLE_RESOLUTIONS = [
    ("AC Supplier", "Power not available", "Inspected AC power supplier unit at site. Replaced blown 15A ceramic primary fuse in rear power entry module, retorqued terminal blocks to 1.8 Nm, and verified stable 230V AC output under 1200W load. Passed 30-minute continuous run test."),
    ("AC Supplier", "Overload tripping", "Found downstream socket bank overloaded by unapproved space heater. Relocated load to secondary branch circuit, reset thermal breaker, and verified normal 4.2A current draw on digital meter."),
    ("Milk Chiller", "Cooling issue", "Condenser coil aluminum fins were thoroughly cleaned of dust and chaff using compressed air. Refrigerant suction pressure checked and topped up with 250g R404A. Chiller achieved 4.0°C holding temperature within 45 minutes."),
    ("Milk Chiller", "Abnormal noise", "Agitator gearbox synthetic oil was depleted. Flushed and refilled gearbox with ISO VG 220 food-grade oil, replaced worn agitator lip seal, and confirmed smooth 28 RPM rotation with zero mechanical noise."),
    ("Pallet Lift", "Incomplete Movement Of The Arm", "Hydraulic fluid level was below minimum mark. Refilled reservoir with ISO VG 32 hydraulic oil, bled air from lift cylinder bleed screw, and adjusted lowering valve seat. Verified 2000 kg rated lift capacity without drift."),
    ("Autopick", "Incomplete Movement Of The Arm", "Robotic arm Joint 2 harmonic drive was bound by plastic strapping debris. Cleared debris, re-greased joint with Molywhite RE00, and performed recalibration homing routine. Completed 100 pick-and-place cycles without fault."),
    ("Voltcore", "Power not available", "UPS inverter fault F-07 was caused by high ambient temperature (42°C). Cleared clogged rear ventilation intake filter and replaced faulty 120mm cooling fan. Inverter returned to normal online operation."),
    ("Paper Shredder", "Generic Test 1", "Cutting cylinder was severely jammed with heavy laminated plastic. Manually extracted jam in reverse mode, applied 15 ml approved shredder oil, and sharpened knives. Tested with 20 sheets 80gsm successfully."),
    ("Exterior Jack", "Generic Test 1", "Lower limit switch bracket was knocked out of alignment by gravel. Realigned bracket, greased acme lead screw with lithium EP2 grease, and confirmed smooth full extension and retraction."),
    ("Asus Celeron", "Power not available", "Workstation power brick DC barrel connector was loose. Replaced DC cable assembly, cleaned heatsink dust blanket, and verified stable boot into Servy diagnostics suite."),
    ("KTM", "Power not available", "Engine failed to start due to fouled spark plug and corroded 12V battery ground wire. Cleaned ground terminal, installed new NGK spark plug, and set idle to 1400 RPM. Started on first crank."),
]

created_calls = 0
for prod_name, ctype, res_text in SAMPLE_RESOLUTIONS:
    prod = products.get(prod_name)
    if not prod:
        continue
    prod_assets = list(Asset.objects.filter(tenant=tenant, product=prod))
    if not prod_assets:
        continue
    for asset in prod_assets[:2]:
        sid = 43000 + created_calls
        sc, _ = ServiceCall.objects.update_or_create(
            tenant=tenant,
            servy_id=sid,
            defaults={
                "call_type": "Service",
                "complaint_type": ctype,
                "complaint_text": f"{ctype} reported on {asset.name} ({asset.model_number}). Resolution completed by technician.",
                "customer": asset.customer,
                "site": asset.site,
                "asset": asset,
                "status": "closed",
                "priority": "normal",
                "contact_name": asset.customer.contact_name if asset.customer else "Site In-Charge",
                "contact_phone": asset.customer.phone if asset.customer else "9999999999",
                "contact_email": asset.customer.email if asset.customer else "info@customer.com",
                "resolution_text": res_text,
                "closed_at": now - timezone.timedelta(days=created_calls * 2 + 1),
                "created_at": now - timezone.timedelta(days=created_calls * 2 + 3),
            }
        )
        created_calls += 1

print(f"Created/updated {created_calls} closed ServiceCalls.")
print("Re-indexing all resolutions into ServiceResolutionIndex...")
reindex_all_resolutions(tenant)

print(f"\n[DONE] Knowledge Base significantly expanded across all categories! Total indexed docs: {KnowledgeDocument.objects.filter(tenant=tenant, index_status='INDEXED').count()}")
