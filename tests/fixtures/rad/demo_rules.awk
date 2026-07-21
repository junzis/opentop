{
#SECTION_1 - Explanatory example rules
#RAD DEMO_REQUIRED - AIRAC 2406 Required central fix
if (DEP == "EHAM" && ARR == "LIRF" && RFL >= 300) { REGLE=REGLE " >D2" }
#RAD DEMO_CALLSIGN - AIRAC 2406 Operator-specific route
if (DEP == "EHAM" && ARR == "LIRF" && CALLSIGN ~/^KLM/) { REGLE=REGLE " >E2" }
#END
}
