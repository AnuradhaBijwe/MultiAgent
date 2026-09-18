flowchart LR
 
    A[Customer Request] --> B[Orchestrator Agent - LLM]
 
    B -->|LLM decides| C[Inventory Agent - LLM]
    B -->|LLM decides| D[Quote Agent - LLM]
    B -->|LLM decides| E[Sales Agent - LLM]
 
    C -->|LLM selects tool| F[Inventory Lookup Tool]
    C -->|LLM selects when shortage exists| G[Supplier Delivery Tool]
 
    D -->|LLM selects tool| H[Quote History Search Tool]
 
    E -->|LLM selects when order can be fulfilled| I[Fulfill Order Tool]
 
    F --> C
    G --> C
    H --> D
    I --> E
 
    C -->|Agent result| B
    D -->|Agent result| B
    E -->|Agent result| B
 
    B --> J[Final Customer Response]