Stage_R is the research repository for Stage R (external-data evaluation), kept fully separate from the strategy factory.
Nothing in this repository imports, reads for compute, or writes to the Monte_Carlo_Strategies factory tree; the only factory contact is reading an API key from a .env path supplied at runtime.
Decisions for this work are recorded in the factory's logs/decision_log.md by the Strategist; this repo holds only tools, tests, and snapshot manifests.
