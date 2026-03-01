# Understanding the x.gx3 File

## What is x.gx3?

The **x.gx3** file in this repository is a **Mitsubishi PLC program file** created by **GX Works3**, which is Mitsubishi's programming software for their newer generation PLCs.

---

## 📋 File Information

| Property | Details |
|----------|---------|
| **File Name** | x.gx3 |
| **File Size** | ~496 KB |
| **File Type** | GX Works3 Project File |
| **Software** | Mitsubishi GX Works3 |
| **Compatible PLCs** | iQ-R, iQ-F, FX5 series |
| **Purpose** | PLC ladder logic program |

---

## 🔧 What Does This File Contain?

### 1. Ladder Logic Program
The main PLC program written in ladder diagram format. This includes:
- Logic for inputs and outputs
- Timers and counters
- Data manipulation
- Communication logic

### 2. Device Memory Configuration
Defines how PLC memory is allocated:
- **D Registers** - Data registers for storing values
- **M Relays** - Internal memory bits
- **X/Y Devices** - Physical I/O points
- **R Registers** - File registers
- **Timers (T)** and **Counters (C)**

### 3. Network Parameters
Communication settings including:
- Ethernet/IP configuration
- MC Protocol settings
- Connection parameters
- Station numbers

### 4. I/O Configuration
Physical input/output mappings:
- Input devices (sensors, switches)
- Output devices (relays, motors)
- Analog modules
- Special function modules

### 5. Documentation
- Comments in ladder logic
- Device labels
- Network diagrams
- Parameter notes

---

## 🔄 Relationship to Dashboard

### How They Work Together

```
┌──────────────────────────────────────────────────────┐
│                  Complete System                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐      ┌──────────┐      ┌─────────┐ │
│  │   x.gx3     │──┐   │   PLC    │  ┌──│Dashboard│ │
│  │  (Program)  │  │   │Hardware  │  │  │ System  │ │
│  └─────────────┘  │   └──────────┘  │  └─────────┘ │
│        │          │         │        │       │      │
│        │          ▼         │        ▼       │      │
│        │    ┌─────────┐    │   ┌─────────┐  │      │
│        │    │Download │    │   │  Read/  │  │      │
│        └───▶│ Program │────┘   │  Write  │◀─┘      │
│             │ to PLC  │        │  Data   │         │
│             └─────────┘        └─────────┘         │
│                                                     │
│   Programming        Execution      Monitoring     │
│   (One-time)         (Continuous)   (Real-time)    │
└─────────────────────────────────────────────────────┘
```

### Workflow

1. **Programming Phase** (Using GX Works3 + x.gx3)
   - Write ladder logic
   - Configure network settings
   - Set up I/O devices
   - Download to PLC

2. **Execution Phase** (PLC Runs Program)
   - PLC executes ladder logic continuously
   - Updates I/O based on program
   - Maintains data in registers

3. **Monitoring Phase** (Using Dashboard)
   - Dashboard connects to running PLC
   - Reads current data values
   - Writes new values
   - Displays real-time status

---

## 🎯 Do You Need This File?

### To Use the Dashboard: **NO** ❌

The dashboard communicates with the **running PLC**, not the program file. You don't need GX Works3 or x.gx3 to use the dashboard if:
- PLC is already programmed
- Network settings are configured
- MC Protocol is enabled
- PLC is running

### To Modify PLC Program: **YES** ✅

You need GX Works3 and x.gx3 if you want to:
- Change ladder logic
- Add new I/O points
- Modify timers/counters
- Update network configuration
- Debug PLC program

---

## 📖 How to Use x.gx3 File

### Requirements

1. **Software**: Mitsubishi GX Works3
   - Download from Mitsubishi website
   - Requires license (may have trial version)

2. **Compatible PLC**: 
   - iQ-R Series (R00, R01, R02, etc.)
   - iQ-F Series (FX5U, FX5UJ, FX5UC)
   - Other supported models

3. **Connection**: 
   - USB cable (PLC to PC)
   - Or Ethernet connection

### Opening the File

1. **Launch GX Works3**
2. **File** → **Open Project**
3. **Navigate to** `AUTOVISION_GUARDIAN` folder
4. **Select** `x.gx3` and click **Open**
5. **View/Edit** ladder diagrams, parameters, etc.

### Downloading to PLC

1. **Open Project** in GX Works3
2. **Online** → **PLC Read** (to read current PLC program)
3. **Or Online** → **PLC Write** (to write this program to PLC)
4. **Select items** to write (program, parameters, etc.)
5. **Execute** and confirm

---

## ⚠️ Important Notes

### Version Control

If you're tracking this file in Git:

**Pros:**
- Backup of PLC program
- Track changes over time
- Share with team members
- Complete system documentation

**Cons:**
- Large file size (~500KB, can grow)
- Binary format (Git can't show diffs)
- May contain sensitive logic

### .gitignore Option

If you prefer not to track in Git, add to `.gitignore`:

```gitignore
# PLC Program Files
*.gx3
*.gx2
```

### Security Considerations

The PLC program may contain:
- Proprietary control logic
- Security configurations
- IP addresses and credentials
- Business process information

**Consider carefully** before sharing publicly!

---

## 🔍 Viewing Without GX Works3

Unfortunately, `.gx3` files are **proprietary** and can only be opened with GX Works3. There's no free viewer or converter available.

**Alternatives:**
- Export ladder diagrams as PDF from GX Works3
- Take screenshots of important logic
- Document the program in separate files

---

## 📊 File Structure (Technical)

The `.gx3` file is a **compressed archive** containing:

```
x.gx3 (ZIP format)
├── PROJECT.GPJ       # Project metadata
├── PROGRAM/          # Ladder logic files
├── PARAMETER/        # Configuration parameters
├── DEVICE/           # Device memory layout
├── LABEL/            # Device labels/comments
└── Other resource files
```

You can rename `x.gx3` to `x.zip` and extract to see contents, but files are in proprietary format.

---

## 🎓 Summary

### Quick Facts

| Question | Answer |
|----------|--------|
| What is x.gx3? | Mitsubishi PLC program file (GX Works3) |
| Do I need it for dashboard? | No, dashboard connects to running PLC |
| Can I delete it? | Yes, if you don't plan to modify PLC program |
| How to open it? | Use Mitsubishi GX Works3 software |
| Can I view without GX Works3? | No, proprietary format |
| Should I keep in repository? | Optional - your choice |

### When You Need It

✅ **Keep if:**
- You need to modify PLC logic
- You want version control of PLC program
- Team needs access to ladder diagrams
- This is a complete system backup

❌ **Remove if:**
- Only using dashboard (don't need to program)
- File is too large for repository
- Contains sensitive/proprietary logic
- PLC programming done elsewhere

---

## 📞 Related Guides

- **[PLC_CONNECTION_GUIDE.md](PLC_CONNECTION_GUIDE.md)** - How to connect PLC to dashboard
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - How to test the system
- **[README.md](README.md)** - Complete documentation

---

**The dashboard works with or without this file!** It only needs a **running, configured PLC** with MC Protocol enabled. 🚀
