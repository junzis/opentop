{
#SECTION_1 - Fixture rules
#RAD RULE1 - AIRAC 2406 Required waypoint
if (DEP ~/AAAA/ && ARR == "DDDD" && RFL > 280) { REGLE=REGLE " >A_B >C" }
#RAD RULE2 - AIRAC 2406 Inactive example
#if (DEP ~/AAAA/) { REGLE=REGLE " -C" }
#RAD RULE3 - AIRAC 2406 Unsupported input is retained
if (UNKNOWN_FN(DEP)) { REGLE=REGLE " -A_B" }
#RAD RULE4 - AIRAC 2406 Flight-level assignment
if (DEP ~/AAAA/ && ARR ~/DDDD/ && $3==0) FL_CONT = 285
#END
}
