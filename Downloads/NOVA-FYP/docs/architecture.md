# NOVA architecture (draft)

```mermaid
flowchart TD
    Camera[Wearable camera] --> Vision[Python / YOLO perception]
    Cane[ESP32 smart cane sensors] --> Bridge[Communication interface]
    Vision --> CAPME[CAPME priority engine]
    Bridge --> CAPME
    App[Android Kotlin app] --> Context[Destination and user context]
    Context --> CAPME
    CAPME --> Audio[Voice alerts]
    CAPME --> Haptics[Wristband / cane haptics]
```

**Deployment assumption:** Python perception/CAPME on Raspberry Pi, sensor firmware on ESP32, native Kotlin on Android. Transport choice and exact power topology remain design decisions.
