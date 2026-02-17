# Smart Scheduler Flask Application - Knowledge Transfer Guide

## Table of Contents

1. [Application Overview](#application-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Configuration & Setup](#configuration--setup)
4. [Core Components](#core-components)
5. [API Endpoints Reference](#api-endpoints-reference)
6. [Key Business Logic](#key-business-logic)
7. [Special Features & Fixes](#special-features--fixes)
8. [Bug History & Solutions](#bug-history--solutions)
9. [Troubleshooting](#troubleshooting)

---

## Application Overview

### Purpose

Smart Scheduler is a Flask-based web application for managing audit resource booking and scheduling in a banking environment. It handles:

- Audit demand suggestions
- Resource availability management
- Booking allocation with distributed hours
- Calendar visualization
- Role-based access control (RBAC)

### Technology Stack

- **Backend**: Flask (Python)
- **Data Source**: Dataiku datasets and managed folders
- **Data Processing**: Pandas
- **Authentication**: OTP-based system
- **Storage**: CSV files in Dataiku managed folders

---

## Architecture & Data Flow

### 1. Data Sources (Dataiku Integration)

**Four Primary Datasets:**

1. **gia_demand_suggestions_latest**
   - Contains unmet audit demands
   - Key columns: audit_issue_number, psid, auditor_days, demand_id, sch_start_date, sch_end_date

2. **gia_persons_base_availability**
   - Resource availability by date
   - Key columns: psid, date, daily_available_hours, is_holiday, is_weekend, is_naa, naa_hours, leave_hours
   - **Critical**: Can contain GROSS (8h) or NET (0.8h post-booking) values depending on sync schedule

3. **fica_audit_final_scheduler_mapped_IN**
   - Audit metadata and PSID mappings
   - Key columns: Audit_ID/audit_issue_number, PSID, Primary_Audit_Issue_Group
   - Used for RBAC filtering

4. **bookings.csv (Managed Folder)**
   - Live booking records
   - Key columns: audit_number, PSID, BookedFrom, BookedTo, allocated_hours, per_date_hours, demand_id

### 2. Caching Strategy

**Global Variables (Module-Level):**

- All datasets loaded once at application startup
- 30-second TTL cache for bookings and availability
- Cache invalidation via `/api/clear_cache` endpoint

**Purpose**: Avoid reloading 7000+ booking rows on every API call (prevents 30-60 sec delays)

---

## Configuration & Setup

### Global Configuration (Lines 18-50)

**Key Settings:**

- `MANAGED_FOLDER_ID`: Dataiku folder for bookings.csv
- `BOOKINGS_FILE_PATH`: "bookings.csv"
- `SUPER_ADMIN_PSIDS`: List of admin PSIDs with full access
- `CACHE_TTL_SECONDS`: 30 seconds
- `COLOR_PALETTE`: 15 distinct colors for audit visualization

**OTP Configuration:**

- `MAPPING_DATASET`: User email mapping
- `PROJECT_KEY`: Dataiku project
- `SCENARIO_ID`: OTP sender scenario

---

## Core Components

### A. Data Loading Functions (Lines 345-530)

#### 1. `load_availability()`

- Loads availability dataset with 30-second cache
- **Critical Fix**: Uses `.astype(float)` to preserve fractional hours (0.8h, 0.5h)
- Handles flexible date parsing (DD-MM-YYYY and YYYY-MM-DD)

#### 2. `load_bookings()`

- Loads bookings from managed folder with cache
- **Key Feature**: Calls `remove_duplicate_bookings()` to merge duplicate rows
- Ensures backward compatibility for new columns
- Returns copy to prevent cache corruption

#### 3. `remove_duplicate_bookings()` (Lines 417-485)

- Merges duplicate bookings with identical: PSID + audit_number + BookedFrom + BookedTo + demand_id
- **Safety**: Includes demand_id to protect legitimate multi-demand bookings
- Sums allocated_hours when merging
- Logs deduplication statistics

#### 4. `parse_date_flexible()` (Helper Function)

- Handles both DD-MM-YYYY and YYYY-MM-DD formats
- Uses `dayfirst=True` for proper parsing

### B. Managed Folder Operations (Lines 395-415)

#### `read_csv_from_folder()`

- Reads CSV from Dataiku managed folder
- Tries multiple delimiters (tab, comma, semicolon)
- Handles encoding issues

#### `write_csv_to_folder()`

- Writes DataFrame to managed folder
- Used for updating bookings.csv

---

## API Endpoints Reference

### 1. Authentication (Lines 570-740)

#### `POST /api/send_otp`

- Generate and send OTP to user email
- Stores in global `otp_store` dictionary
- Triggers Dataiku scenario for email sending

#### `POST /api/verify_otp`

- Validates OTP against stored value
- Returns success/failure

### 2. Audit Management (Lines 1972-2178)

#### `GET /api/audits`

- Returns list of audits filtered by user's Primary_Audit_Issue_Group
- **Critical Fix**: Handles both 'Audit_ID' and 'audit_issue_number' column names
- Merges demand data with audit metadata

#### `GET /api/audits/<audit_number>`

- Returns details for specific audit
- Used for drill-down views

### 3. Booking Operations (Lines 2179-3114)

#### `GET /bookings` and `GET /bookings/<audit_number>`

- Retrieve all bookings or bookings for specific audit
- Returns booking list without sensitive data

#### `POST /bookings` (add_booking)

- **Most Complex Function** - handles new booking creation
- Steps:
  1. Parse booking dates
  2. Check date clashes
  3. Load availability for date range
  4. Calculate distributed hours per day
  5. Update per_date_hours string format
  6. Save to bookings.csv
  7. Clear related caches

#### `DELETE /bookings`

- Delete booking by unique combination of fields
- Updates managed folder

#### `POST /bookings/delete`

- Batch delete bookings by demand_id
- Used for removing multiple related bookings

#### `GET /download_all_bookings` and `POST /download_selected_bookings`

- Export bookings to CSV
- **Critical Fix**: Standardizes dates to DD-MM-YYYY format before export

### 4. Resource Availability (Lines 1279-1970)

#### `GET /api/auditors/<audit_number>`

- Returns available auditors for specific audit
- Calculates booking status (Met/Unmet/Partially Met)
- Computes allocated vs required days

#### `GET /api/auditor_booking_status/<audit_number>/<int:psid>`

- Detailed booking status for specific auditor on audit
- Returns allocated days, remaining capacity, booking details
- **Critical Fix**: Initializes `allocated_days_calculated` before try block to avoid UnboundLocalError

#### `POST /api/check_auditor_availability`

- Checks if auditor has sufficient hours for booking
- Returns available hours by date
- **Key Feature**: Detects NET vs GROSS availability values

### 5. Calendar Visualization (Lines 3420-3825)

#### `GET /api/calendar_colors/<audit_number>/<int:psid>`

- **Most Complex Visualization Function**
- Generates color-coded calendar for date range
- Color Logic:
  - **Brown**: NAA (8h unavailable)
  - **Light-blue**: Weekend/Holiday (0h available)
  - **Green**: 8h remaining
  - **Yellow-dark**: 7h remaining
  - **Yellow-medium**: 5h remaining
  - **Yellow-light**: 3h remaining
  - **Yellow-very-light**: >0h remaining (includes 0.8h, 0.5h)
  - **Grey**: Out of availability window

**Critical Features:**

1. Loads bookings and calculates distributed hours by date
2. Creates availability maps from dataset
3. **NET vs GROSS Detection** (Lines 3703-3722):
   - If `available_hours < booked_hours`, treats as NET (post-booking)
   - Uses value directly without subtracting bookings again
   - Prevents double-accounting bug
4. Iterates through date range applying color rules
5. Returns JSON with color, available_hours, booked_hours, remaining_hours

#### `GET /api/color_legend`

- Returns color palette explanation
- Used for UI legend display

#### `POST /api/clear_cache`

- Manually clears all caches (availability, bookings)
- Forces fresh data reload from Dataiku
- Use after updating datasets

### 6. Demand Suggestions (Lines 1064-1278)

#### `GET /api/demands`

- Returns demand suggestions filtered by user group
- Status tracking: Met/Unmet

#### `GET /api/demand_summary/<audit_number>`

- Aggregated demand summary for audit
- Shows total required vs allocated days

### 7. Booking Calendar View (Lines 3115-3419)

#### `GET /api/bookings_for_calendar/<audit_number>`

- Returns all bookings for audit calendar view
- Includes auditor details merged from multiple datasets

#### `GET /api/bookings/<int:psid>/<audit_number>`

- Returns bookings for specific PSID and audit
- Used for detail panels

---

## Key Business Logic

### 1. Distributed Hours Calculation

**Concept**: When booking 10 days for 5h total, distribute evenly across available days

**Implementation** (in `add_booking` function):

```
Daily Hours = Total Hours / Number of Available Days
Skips: Weekends, Holidays, NAA days
```

**Storage Format**:

```
per_date_hours: "2026-06-08:0.5;2026-06-09:0.5;2026-06-10:0.5"
book_dates_list: "2026-06-08;2026-06-09;2026-06-10"
```

### 2. Availability Calculation Logic

**Two Scenarios:**

**A. Before Midnight Sync** (GROSS availability):

```
PSID has: 8h available
Booked: 7.2h
Remaining = 8h - 7.2h = 0.8h ✓
```

**B. After Midnight Sync** (NET availability):

```
Availability file updated: 0.8h (net remaining)
Booked: 7.2h still in bookings.csv
Code detects: 0.8h < 7.2h → Use 0.8h directly (no subtraction)
Remaining = 0.8h ✓
```

### 3. Date Clash Detection

**Function**: `check_date_clash()`

- Checks if PSID already booked in overlapping date range
- Returns clash status and conflicting audit number
- Prevents double-booking

### 4. Role-Based Access Control (RBAC)

**Implementation**:

- Filters audits by `Primary_Audit_Issue_Group`
- Super admin PSIDs bypass filtering
- Applied in: `/api/audits`, `/api/demands`, `/api/auditors`

### 5. Booking Status Calculation

**Three States**:

1. **Met**: `allocated_days >= required_days`
2. **Unmet**: `allocated_days == 0`
3. **Partially Met**: `0 < allocated_days < required_days`

---

## Special Features & Fixes

### 1. Fractional Hour Support (CRITICAL)

**Problem**: Original code used `.astype(int)` truncating 0.8h → 0h
**Fix**: Changed to `.astype(float)` in `load_availability()` (Line 367)
**Impact**: Allows booking of 0.8h, 0.5h, 0.3h slots

### 2. NET vs GROSS Availability Detection (CRITICAL)

**Problem**: After midnight sync, availability shows 0.8h (NET) but code subtracts 7.2h bookings again
**Fix**: Added detection logic (Lines 3713-3722)

```
If available_hours < booked_hours:
    Use available_hours directly (NET mode)
Else:
    remaining = available_hours - booked_hours (GROSS mode)
```

### 3. Duplicate Booking Removal

**Problem**: Duplicate bookings inflating booked hours
**Solution**: `remove_duplicate_bookings()` merges identical bookings
**Safety**: Includes demand_id to protect multi-demand bookings
**Stats**: Removed 413 duplicate rows from live system

### 4. Dynamic Column Detection

**Problem**: Dataset column names changed ('Audit_ID' → 'audit_issue_number')
**Fix**: `get_audits()` checks both column names dynamically (Line 1983)

### 5. Date Format Standardization

**Problem**: Mixed YYYY-MM-DD and DD-MM-YYYY in exports
**Fix**: Download endpoints standardize to DD-MM-YYYY before CSV export

### 6. SettingWithCopyWarning Fix

**Problem**: Pandas warning when adding date_str column
**Fix**: Added `.copy()` when filtering DataFrames (Line 3648)

### 7. UnboundLocalError Fix

**Problem**: `allocated_days_calculated` referenced before assignment in exception cases
**Fix**: Initialize to 0 before try blocks (Lines 1581, 1755)

---

## Bug History & Solutions

### Production Bugs We've Faced (Feb 2026)

This section documents real production issues encountered and their resolutions. Use this as a reference when similar symptoms appear.

---

### Bug #1: 3260 Audit Suggestions Not Appearing in UI

**Severity**: High (Data Visibility)  
**Date Reported**: Feb 2026

**Symptoms:**

- Base file shows 3260 audit records
- UI displays only partial audits or empty list
- No errors in logs

**Root Cause:**

- Dataset column name changed from `Audit_ID` to `audit_issue_number`
- Code hardcoded column name as `'Audit_ID'`
- KeyError silently caught, returning empty results

**Investigation Method:**

```
1. Check dataset columns: df.columns
2. Search for hardcoded 'Audit_ID' references
3. Compare with actual dataset structure
```

**Solution Applied:**

- Added dynamic column detection in `get_audits()` function (Line 1983)
- Code now checks: `'audit_issue_number' if 'audit_issue_number' in df.columns else 'Audit_ID'`
- Backward compatible with old datasets

**Prevention:**

- Always use dynamic column checking for external datasets
- Log column names on data load for debugging
- Add dataset schema validation at startup

---

### Bug #2: 0.8h Availability Showing as 0hr (Gray/Unavailable)

**Severity**: Critical (Business Impact)  
**Date Reported**: Feb 2026

**Symptoms:**

- User has 0.8h available per day
- Calendar shows 0hr (gray, unselectable)
- Cannot book fractional hours
- Logs show `available_hours=0`

**Root Cause (Multiple Issues):**

**Issue 2a: Integer Truncation**

- Line 365: `daily_available_hours.astype(int)` truncated 0.8 → 0
- All fractional hours lost

**Issue 2b: Availability Threshold**

- Line 3598: `if remaining_hours >= 1` prevented fractional hour selection
- 0.8h treated as unusable

**Issue 2c: Date Key Format Mismatch**

- `availability_map` used date objects as keys
- Lookup used string keys `'2026-06-08'`
- Mismatch returned 0 for existing dates

**Investigation Method:**

```
1. Add diagnostic logs: print(daily_available_hours, type(daily_available_hours))
2. Check availability_map keys: print(list(availability_map.keys())[:5])
3. Verify data type conversion: df['daily_available_hours'].dtype
4. Test with known dates: availability_map.get('2026-06-08', 'NOT FOUND')
```

**Solution Applied:**

- Changed `astype(int)` → `astype(float)` (Line 367)
- Changed threshold `>= 1` → `> 0` (Lines 3598, 3630)
- Standardized all date keys to strings: `date_str = date.strftime('%Y-%m-%d')`
- Added `dayfirst=True` for DD-MM-YYYY parsing

**Prevention:**

- Use `.astype(float)` for any hour/capacity calculations
- Never use `>= 1` threshold for availability checking
- Always standardize date formats (string vs object) across entire codebase
- Add data type validation after loading datasets

---

### Bug #3: Double-Accounting Bug (NET vs GROSS Availability)

**Severity**: Critical (Data Integrity)  
**Date Reported**: Feb 2026

**Symptoms:**

- After midnight sync, 0.8h shows as 0h (gray)
- Before sync, same date shows properly
- Math: `0.8h (available) - 7.2h (booked) = -6.4h → 0h`

**Root Cause:**

- Dataiku pipeline updates availability file at midnight
- After sync: availability file contains NET remaining hours (8h - 7.2h = 0.8h)
- Code assumes GROSS capacity and subtracts bookings again
- Result: `0.8h - 7.2h = negative → 0h`

**Business Context:**

```
Timeline:
09:00 AM - Resource has 8h available
10:00 AM - Booked 7.2h, remaining 0.8h shows correctly
00:00 AM - Midnight sync updates availability to 0.8h (NET)
08:00 AM - Same date now shows 0h (double subtraction)
```

**Investigation Method:**

```
1. Compare availability values before/after midnight
2. Check if available_hours < booked_hours (indicates NET)
3. Add logs: "available_hours={X}, booked_hours={Y}, raw_remaining={X-Y}"
4. Verify booking records haven't changed overnight
```

**Solution Applied:**

- Added NET vs GROSS detection logic (Lines 3713-3722)
- If `0 < available_hours < booked_hours`: Use available_hours directly (NET mode)
- Else: Calculate `remaining_hours = available_hours - booked_hours` (GROSS mode)
- Logs now show: "NET AVAILABILITY detected, using {X}h directly"

**Prevention:**

- Document whether availability dataset contains GROSS or NET values
- Add metadata flag in availability dataset: `is_net_capacity` column
- Consider keeping availability at GROSS always, handle NET in reporting layer
- Add validation: Alert if available < booked for multiple dates

---

### Bug #4: Duplicate Bookings Consuming 2x Capacity

**Severity**: High (Data Quality)  
**Date Reported**: Feb 2026

**Symptoms:**

- Same booking appears multiple times in bookings.csv
- Distributed hours doubled: 0.5h + 0.5h = 1.0h total
- All capacity consumed even though only one actual booking
- Example: PSID+audit+dates identical, but 2 rows in CSV

**Root Cause:**

- Race condition in concurrent booking requests
- No unique constraint on bookings.csv
- Dataiku managed folder append operations not atomic

**Investigation Method:**

```
1. Group bookings by: PSID, audit_number, BookedFrom, BookedTo, demand_id
2. Count duplicates: grouped.filter(lambda x: len(x) > 1)
3. Check distributed_hours_by_date for inflated values
4. Verify demand_id matches (rule out legitimate multi-demand)
```

**Solution Applied:**

- Added `remove_duplicate_bookings()` function (Lines 417-485)
- Groups by: PSID + audit_number + BookedFrom + BookedTo + demand_id
- Sums allocated_hours when merging
- Runs automatically on every cache refresh (30 seconds)
- Production result: Removed 413 duplicate rows from 7492 total

**Prevention:**

- Add unique constraint validation before saving bookings
- Implement optimistic locking with version/timestamp column
- Consider moving to proper database (SQLite/PostgreSQL)
- Add client-side request deduplication (disable button after click)

---

### Bug #5: Inconsistent Date Formats in Downloaded CSV

**Severity**: Medium (UX/Data Quality)  
**Date Reported**: Feb 2026

**Symptoms:**

- Some dates show as `2026-06-08` (YYYY-MM-DD)
- Other dates show as `08-06-2026` (DD-MM-YYYY)
- Same file has mixed formats
- Excel/Notepad users confused

**Root Cause:**

- Bookings stored with mixed date formats in Dataiku
- Download endpoints used `.to_csv()` without format standardization
- Pandas preserves original string format from source

**Investigation Method:**

```
1. Open downloaded CSV in text editor
2. Check BookedFrom/BookedTo columns for format patterns
3. Trace back to data source format
4. Verify parse_date_flexible() handling
```

**Solution Applied:**

- Added date standardization before CSV export (Lines 2206-2217, 2238-2249)
- Parse with `pd.to_datetime(dayfirst=True, errors='coerce')`
- Format all dates as: `dt.strftime('%d-%m-%Y')` (DD-MM-YYYY)
- Applied to both `/download_all_bookings` and `/download_selected_bookings`

**Prevention:**

- Standardize date format at data entry (POST /bookings)
- Store dates in ISO format (YYYY-MM-DD) internally
- Convert to regional format only at display/export layer
- Add date format validation in data pipeline

---

### Bug #6: UnboundLocalError Crashing Booking Status API

**Severity**: Medium (Stability)  
**Date Reported**: Feb 2026

**Symptoms:**

- API call `/api/auditor_booking_status/<audit>/<psid>` returns 500 error
- Error: "local variable 'allocated_days_calculated' referenced before assignment"
- Only occurs for specific auditor-audit combinations

**Root Cause:**

- Variable `allocated_days_calculated` defined inside try block
- Exception thrown before assignment (e.g., date parsing error)
- Exception handler references undefined variable
- Python raises UnboundLocalError

**Investigation Method:**

```
1. Search for variable name in exception traceback
2. Check if variable assignment happens after potential exception point
3. Review exception handler code for variable usage
4. Test with problematic PSID/audit combination
```

**Solution Applied:**

- Initialize `allocated_days_calculated = 0` before try block (Lines 1581, 1755)
- Ensures variable always defined even if exception occurs
- Safe default value (0 days allocated)

**Prevention:**

- Initialize all variables at function start with safe defaults
- Use `variable = variable if 'variable' in locals() else 0` in exception handlers
- Add unit tests for exception paths
- Use linters (pylint) to catch uninitialized variables

---

### Bug #7: SettingWithCopyWarning in Calendar Colors

**Severity**: Low (Warning/Performance)  
**Date Reported**: Feb 2026

**Symptoms:**

- Warning in logs: "SettingWithCopyWarning: A value is trying to be set on a copy..."
- No immediate functionality impact
- Potential silent data corruption risk

**Root Cause:**

- Pandas DataFrame filtered without `.copy()`
- Modifying filtered view affects original cached DataFrame
- Pandas warns about ambiguous assignment

**Investigation Method:**

```
1. Search logs for "SettingWithCopyWarning"
2. Identify line number from warning
3. Check if DataFrame was filtered then modified
4. Verify if cached data is affected
```

**Solution Applied:**

- Added `.copy()` when filtering DataFrames (Line 3648)
- Example: `emp_availability = df[df['psid'] == psid].copy()`
- Creates independent copy, safe to modify

**Prevention:**

- Always use `.copy()` after filtering if you'll modify the result
- Or use `.loc[]` for in-place modifications
- Enable Pandas warnings in development: `pd.options.mode.chained_assignment = 'raise'`
- Run code review checklist for DataFrame operations

---

### Key Takeaways for Future Debugging

**1. Add Diagnostic Logging Early**

```
Pattern: [DEBUG] PSID {X} {date}: available={A}, booked={B}, remaining={C}
Benefit: See exact values causing issues
```

**2. Never Assume Data Types**

```
Always check: df['column'].dtype after loading
Always convert: .astype(float) for numeric values
```

**3. Validate Date Formats Consistently**

```
Use: pd.to_datetime(dayfirst=True, errors='coerce')
Store: YYYY-MM-DD strings internally
Display: DD-MM-YYYY for users
```

**4. Cache Invalidation Strategy**

```
After dataset updates: POST /api/clear_cache
Or wait: 30 seconds for auto-refresh
Verify: Check logs for "[LOAD] Loaded X rows"
```

**5. Test with Edge Cases**

```
- Fractional hours: 0.8h, 0.5h, 0.3h
- Zero availability: 0h
- Over-allocation: booked > available
- Date boundaries: month-end, year-end
- Duplicate records
- Missing data (NaN, None)
```

**6. Common Debug Commands**

```python
# Check data types
print(df.dtypes)

# Inspect problematic rows
print(df[df['psid'] == 1637353])

# Check for duplicates
print(df.duplicated(subset=['key']).sum())

# Verify date parsing
print(pd.to_datetime('08-06-2026', dayfirst=True))

# Test availability map
print(list(availability_map.keys())[:10])
print(availability_map.get('2026-06-08', 'NOT FOUND'))
```

---

## Troubleshooting

### Common Issues

#### 1. "0.8h showing as 0h (gray/unavailable)"

**Diagnosis**:

- Check logs: `[DEBUG] PSID X 2026-06-08: available_hours=?, booked_hours=?`
- If `available_hours=0`: Availability file missing data for PSID/date
- If `remaining=-6.4`: NET availability detection not working

**Solution**:

- Call `POST /api/clear_cache` to reload availability
- Verify PSID exists in availability dataset
- Check if NET detection triggered in logs

#### 2. "3260 audits not showing in UI"

**Diagnosis**: Column name mismatch
**Solution**: Code now handles both 'Audit_ID' and 'audit_issue_number'

#### 3. "Duplicate bookings consuming capacity"

**Diagnosis**: Check `distributed_hours_by_date` logs
**Solution**: Deduplication runs automatically every 30 seconds

#### 4. "Cache not refreshing"

**Solution**:

- Wait 30 seconds for auto-refresh
- OR call `POST /api/clear_cache`
- Check `CACHE_TTL_SECONDS` setting

### Diagnostic Endpoints

1. **Check if PSID exists in availability:**
   - Logs show: `[CHECK] PSID X exists in availability: True/False`

2. **See bookings for date range:**
   - Logs show: `[DEBUG] PSID X has Y bookings overlapping Jun 8-22`

3. **View availability values:**
   - Logs show: `[DEBUG] PSID X Jun 8-22 availability: [(date, hours)]`

4. **Monitor deduplication:**
   - Logs show: `[DEDUPE] Removed X duplicate bookings`

---

## Key Points for KT Session

### 1. Data Flow Summary

```
Dataiku Datasets → Cache (30s) → API Endpoints → Frontend
                         ↓
                  Managed Folder (bookings.csv)
```

### 2. Critical Functions to Understand

1. `load_bookings()` - Entry point for all booking operations
2. `add_booking()` - Complex distributed hours logic
3. `get_calendar_colors_for_employee()` - Visualization engine
4. `check_auditor_availability()` - Capacity checking

### 3. Most Common Developer Tasks

- Adding new API endpoint: Follow pattern in lines 2179+
- Modifying booking logic: Update `add_booking()` function
- Changing availability calculation: Modify calendar colors function
- Adding new dataset: Update module-level loading section

### 4. Testing Checklist

- [ ] Fractional hours (0.8h) show as selectable (yellow-very-light)
- [ ] Download CSV has consistent DD-MM-YYYY dates
- [ ] Duplicate bookings don't inflate hours
- [ ] NET vs GROSS availability handled correctly
- [ ] Cache clears properly with endpoint
- [ ] RBAC filters audits by user group

### 5. Deployment Notes

- All datasets must be available at startup
- Managed folder must have write permissions
- OTP scenario must be configured
- Super admin PSIDs list must be updated for production

---

## Quick Reference - Line Numbers

| Component           | Lines     |
| ------------------- | --------- |
| Configuration       | 18-50     |
| Data Loading        | 345-530   |
| Duplicate Removal   | 417-485   |
| Authentication      | 570-740   |
| Audits API          | 1972-2178 |
| Bookings CRUD       | 2179-3114 |
| Availability Checks | 1279-1970 |
| Calendar Colors     | 3420-3825 |
| Cache Management    | 3810-3835 |

---

## KT Session Script - Code Walkthrough

### How to Use This Script

This is a conversational walkthrough you can use to explain the code line-by-line to new developers. Open `app.py` side-by-side and follow along.

---

### **Session Start: Introduction (5 mins)**

**"Welcome to Smart Scheduler! Let me walk you through the entire codebase. Since you've already seen the UI and understand the flow, we'll focus on how the code actually works behind the scenes.**

**This is a Flask application with about 5000+ lines. Don't worry - we'll go through it logically, and I'll explain the important parts. Open app.py in your editor so you can follow along.**

**Let's start from the top..."**

---

### **Part 1: Imports and Configuration (Lines 1-70)**

**"First, let's look at our imports at the very top of the file:**

**Lines 1-16: Standard Python Libraries**

- Flask for the web framework
- Pandas for data manipulation - we use this HEAVILY
- Dataiku for connecting to our datasets
- openpyxl for Excel exports
- datetime for date handling

**Lines 18-50: Configuration Section**

**This is critical. Let me explain each config variable:**

**`MANAGED_FOLDER_ID = "EcobzUlG"`**

- This is where bookings.csv lives in Dataiku
- Think of it as our database - every booking write goes here

**`BOOKINGS_FILE_PATH = "bookings.csv"`**

- The actual filename in that folder
- Contains all live bookings

**`SUPER_ADMIN_PSIDS = [1406088, 1666904, ...]`**

- These PSIDs can see ALL audits and bookings
- Everyone else sees only their group's data (RBAC)

**`CACHE_TTL_SECONDS = 30`**

- We cache bookings and availability for 30 seconds
- Why? Because loading 7000+ bookings every API call takes 30-60 seconds!
- This one setting made the app 100x faster

**`COLOR_PALETTE = [...]`**

- Colors for the calendar UI
- Each audit type gets a unique color automatically

**Important question to ask: "Do you understand why we need caching? Imagine if every calendar click took 60 seconds to load..."**

---

### **Part 2: Global Dataset Loading (Lines 80-150)**

**"Now, here's something interesting. Around line 80-150, we load all datasets at MODULE level - meaning they load ONCE when the app starts, not on every request.**

**Let me show you:"**

**`demand_suggestions_df = get_demand_suggestions_df()`**

- Loads ALL demand suggestions once
- Contains unmet audit requirements
- Used for matching resources to audits

**`availability_df = get_availability_df()`**

- Resource availability by date
- This is KEY - tells us who's available on which days

**`audit_df = get_audit_df()`**

- Audit metadata
- Used for filtering by group (RBAC)

**"Why load at module level? Performance! These datasets don't change often, so we load them once and reuse. Only bookings.csv changes frequently, so that has a 30-second cache."**

**Pop quiz for them: "What happens if availability data changes in Dataiku? How do we refresh it?"**
_(Answer: Restart Flask app or wait for next deployment)_

---

### **Part 3: The Caching System (Lines 345-380)**

**"Let's look at one of the most important functions: `load_availability()` around line 345.**

**Walk through the code:**

```
def load_availability():
    global AVAILABILITY_DATAFRAME_CACHE

    # Check if cache is valid
    current_time = time.time()
    if cache is fresh (< 30 seconds old):
        return cached copy

    # Cache expired - reload from Dataiku
    df = get_availability_df()

    # CRITICAL: Convert to float, not int
    df['daily_available_hours'] = ...astype(float)
```

**"Stop here. This line 367 is CRITICAL. See that `.astype(float)`? Originally it was `.astype(int)`. Do you know what happened?"**

_(Wait for answer)_

**"Right - it truncated 0.8 hours to 0! Users couldn't book fractional hours. We spent 2 days debugging this. Always use float for hours/capacity."**

**"Now look at the caching logic:**

- Check timestamp
- If < 30 seconds old, return cached copy
- Else reload from Dataiku and update cache
- ALWAYS return `.copy()` to prevent modifying cached data

**"This pattern repeats in `load_bookings()`. Same logic, different data."**

---

### **Part 4: The Duplicate Booking Remover (Lines 417-485)**

**"Around line 417, there's a function called `remove_duplicate_bookings()`. Let me tell you a story about why this exists..."**

**"In production, we found bookings appearing TWICE in bookings.csv. Same PSID, same audit, same dates - but two rows. This meant:**

- 0.5h booking counted as 1.0h
- People showed fully booked when they had capacity
- We removed 413 duplicate rows from live data!

**Walk through the logic:**

```
Group by: PSID + audit_number + BookedFrom + BookedTo + demand_id

Why demand_id?
- Same person CAN be on same audit for different demands
- We only merge TRUE duplicates

When merging:
- Sum allocated_hours (0.5 + 0.5 = 1.0)
- Keep first value for other fields
```

**"This runs automatically every 30 seconds when bookings reload. Silent protection against duplicates."**

**Question: "Why not fix the root cause - prevent duplicates at write time?"**
_(Good discussion point - talk about race conditions, no DB constraints on CSV, etc.)_

---

### **Part 5: Booking Creation - The Complex One (Lines 2246-2750)**

**"Now we get to the most complex function: `add_booking()` around line 2246. This is where bookings actually get created. Grab a coffee, this one's dense."**

**"When a user clicks 'Book' on the UI, this is what happens:**

**Step 1: Parse Input (Lines 2250-2260)**

```python
psid = int(data['psid'])
start_date = pd.to_datetime(data['booked_from'], dayfirst=True)
end_date = pd.to_datetime(data['booked_to'], dayfirst=True)
```

**"Always use `dayfirst=True` for DD-MM-YYYY format. Otherwise '06-08-2026' becomes August 6 instead of June 8!"**

**Step 2: Check Date Clashes (Line 2265)**

```python
is_ok, clash_audit = check_date_clash(psid, start_date, end_date)
```

**"This checks if the person is already booked during these dates. No double-booking allowed."**

**Step 3: Load Availability (Lines 2275-2290)**

**"Here's where it gets interesting. We load availability for ONLY the date range:"**

```python
availability_in_range = availability_df[
    (availability_df['psid'] == psid) &
    (availability_df['date'] >= start_date) &
    (availability_df['date'] <= end_date)
]
```

**"Then we filter OUT holidays, weekends, and NAA days. Why? You can't book someone on days they're unavailable!"**

**Step 4: Distributed Hours Calculation (Lines 2300-2325)**

**"This is THE KEY BUSINESS LOGIC. Let me explain with an example:**

**User wants to book:**

- 10 days (June 1-10)
- 5 hours total

**But in those 10 days:**

- 2 are weekends (skip)
- 1 is holiday (skip)
- 7 are working days

**Calculation:**

```
daily_hours = 5 hours / 7 available days = 0.71 hours per day
```

**"So we spread 5 hours across 7 days. This is called DISTRIBUTED HOURS."**

**Walk through the code:**

```python
# Count available days (excluding weekends/holidays)
available_days = len(availability_in_range)

# Calculate hours per day
daily_distributed_hours = allocated_hours / available_days

# Create per-date hour mapping
per_date_hours = {}
for each available day:
    per_date_hours[date] = daily_distributed_hours
```

**Step 5: Format per_date_hours String (Lines 2330-2345)**

**"We store this as a string: `'2026-06-01:0.71;2026-06-02:0.71;2026-06-03:0.71'`**

**Why a string? Because bookings.csv is a CSV file, not a database. We can't store objects."**

**Step 6: Create Booking Record (Lines 2350-2380)**

**"Finally, we create a new row:"**

```python
new_booking = pd.DataFrame([{
    'audit_number': ...,
    'PSID': psid,
    'BookedFrom': start_date.strftime('%d-%m-%Y'),
    'BookedTo': end_date.strftime('%d-%m-%Y'),
    'allocated_hours': total_hours,
    'daily_distributed_hours': hours_per_day,
    'per_date_hours': '2026-06-01:0.71;2026-06-02:0.71',
    'book_dates_list': '2026-06-01;2026-06-02',
    ...
}])
```

**Step 7: Save to Managed Folder (Lines 2390-2400)**

**"We append to bookings.csv and write back:"**

```python
bookings = pd.concat([existing_bookings, new_booking])
write_csv_to_folder(bookings, BOOKINGS_FILE_PATH)
```

**"And we MUST clear the cache so next API call sees the new booking!"**

**Question: "What happens if we forget to clear cache?"**
_(Answer: New booking won't show up for 30 seconds!)_

---

### **Part 6: Calendar Colors - The Visualization Engine (Lines 3420-3825)**

**"Now let's look at how we generate the colorful calendar. Function: `get_calendar_colors_for_employee()` around line 3420."**

**"This function does ONE thing: For a given PSID and date range, return a color for each day showing availability."**

**The Color Code:**

- **Green**: 8h available
- **Yellow-dark**: 7h available
- **Yellow-medium**: 5h available
- **Yellow-light**: 3h available
- **Yellow-very-light**: Any remaining capacity (even 0.8h!)
- **Light-blue**: Holiday/Weekend (0h)
- **Brown**: NAA - Not Available (8h blocked)
- **Grey**: Outside availability window

**Walk through the logic:**

**Step 1: Load Data (Lines 3430-3475)**

```
1. Load availability for this PSID
2. Load bookings for this PSID
3. Define date range (availability window + 30 days buffer)
```

**Step 2: Calculate Booked Hours by Date (Lines 3480-3620)**

**"This is complex. We iterate through all bookings and parse the per_date_hours string:"**

```python
per_date_hours = "2026-06-08:7.2;2026-06-09:7.2"

Parse this into:
distributed_hours_by_date = {
    '2026-06-08': 7.2,
    '2026-06-09': 7.2
}
```

**"If multiple bookings on same date, we SUM the hours:"**

```python
distributed_hours_by_date['2026-06-08'] =
    existing_value + new_booking_value
```

**Step 3: Build Availability Maps (Lines 3645-3690)**

**"We create quick lookup dictionaries:"**

```python
availability_map = {'2026-06-08': 8.0, '2026-06-09': 0.8, ...}
holiday_map = {'2026-06-04': True, ...}
weekend_map = {'2026-06-05': True, ...}
naa_map = {'2026-10-01': True, ...}
```

**"This makes lookups O(1) instead of O(n). Performance optimization!"**

**Step 4: THE CRITICAL BUG FIX (Lines 3713-3722)**

**"Stop. This section has a bug fix that took us 3 days to find. Let me explain the problem:**

**Original code:**

```python
available_hours = availability_map.get(date)
booked_hours = distributed_hours_by_date.get(date)
remaining = available_hours - booked_hours
```

**"Seems logical, right? But there's a gotcha. At midnight, Dataiku updates the availability file with NET values (after subtracting existing bookings)."**

**Timeline:**

- 10 AM: Person has 8h, we book 7.2h, shows 0.8h remaining ✓
- Midnight: Dataiku sync updates availability to 0.8h (NET)
- 8 AM next day: Code calculates 0.8h - 7.2h = -6.4h → Shows 0h! ✗

**"This is DOUBLE SUBTRACTION. We're subtracting bookings from already-subtracted availability!"**

**The Fix:**

```python
if booked_hours > 0 and 0 < available_hours < booked_hours:
    # NET availability detected!
    remaining_hours = available_hours  # Use directly
else:
    # GROSS availability
    remaining_hours = available_hours - booked_hours
```

**"We detect when availability is NET (available < booked) and skip the subtraction. Problem solved!"**

**Question: "How would YOU have debugged this?"**
_(Good discussion: Add logs, compare before/after midnight, check if bookings changed, etc.)_

**Step 5: Assign Colors (Lines 3725-3805)**

**"Finally, we loop through each date and assign colors based on rules:"**

```python
Priority order:
1. Brown if NAA (8h)
2. Light-blue if weekend/holiday
3. Light-blue if 0h remaining
4. Color grades for remaining hours
5. Grey if outside availability window
```

**"The function returns JSON:"**

```json
{
  "availability_colors": {
    "2026-06-08": {
      "color": "yellow-very-light",
      "available_hours": 8.0,
      "booked_hours": 7.2,
      "remaining_hours": 0.8
    }
  }
}
```

**"The frontend uses this to render the calendar!"**

---

### **Part 7: Other Important APIs (Quick Tour)**

**"Let me quickly show you other key endpoints:**

**`GET /api/audits` (Line 1972)**

- Returns list of audits user can see
- Filters by Primary_Audit_Issue_Group (RBAC)
- Bug we fixed: Column name was hardcoded as 'Audit_ID' but dataset had 'audit_issue_number'
- Now it checks both dynamically!

**`GET /api/auditors/<audit>` (Line 1279)**

- Shows available resources for an audit
- Calculates Met/Unmet/Partially Met status
- Compares allocated_days vs required_days

**`GET /api/check_auditor_availability` (Line 1620)**

- Checks if resource has capacity for new booking
- Returns available hours by date
- Used before showing booking form

**`GET /download_all_bookings` (Line 2192)**

- Exports bookings to CSV
- Bug we fixed: Dates were mixed YYYY-MM-DD and DD-MM-YYYY
- Now we standardize to DD-MM-YYYY before export

**`POST /api/clear_cache` (Line 3810)**

- Force reload of all cached data
- Use after updating Dataiku datasets
- Clears both availability and bookings cache

---

### **Part 8: Common Bugs and Gotchas**

**"Let me share some bugs we hit so you don't repeat them:**

**Bug 1: Integer Truncation**

```python
# WRONG
df['hours'] = df['hours'].astype(int)  # 0.8 becomes 0!

# RIGHT
df['hours'] = df['hours'].astype(float)  # Preserves 0.8
```

**Bug 2: Date Key Mismatch**

```python
# WRONG
availability_map = df.set_index('date')  # Keys are date objects
value = availability_map.get('2026-06-08')  # Lookup with string = FAIL

# RIGHT
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
availability_map = df.set_index('date_str')  # Keys are strings
value = availability_map.get('2026-06-08')  # Works!
```

**Bug 3: Variable Unbound**

```python
# WRONG
try:
    allocated_days = calculate()
except:
    pass
return allocated_days  # ERROR if exception!

# RIGHT
allocated_days = 0  # Initialize with default
try:
    allocated_days = calculate()
except:
    pass
return allocated_days  # Always safe
```

**Bug 4: Modifying Cached Data**

```python
# WRONG
bookings = load_bookings()
bookings['new_col'] = 'value'  # Modifies cache!

# RIGHT
bookings = load_bookings().copy()  # Safe to modify
bookings['new_col'] = 'value'
```

**Bug 5: Date Parsing**

```python
# WRONG
date = pd.to_datetime('08-06-2026')  # Thinks month=08, day=06

# RIGHT
date = pd.to_datetime('08-06-2026', dayfirst=True)  # DD-MM-YYYY
```

---

### **Part 9: Performance Considerations**

**"Let's talk about why this app is fast:**

**1. Module-Level Dataset Loading**

- Datasets load once at startup, not per request
- Saves 10-30 seconds per API call

**2. 30-Second Cache**

- Bookings cached for 30 seconds
- Prevents reading 7000 rows from Dataiku on every click
- Reduced load time from 60s to <1s

**3. Dictionary Lookups**

- Convert DataFrames to dicts for O(1) lookups
- Example: `availability_map = df.set_index('date').to_dict()`

**4. Copy-on-Return**

- Always return `.copy()` from cache
- Prevents accidental cache modification

**Question: "What would happen if we removed caching?"**
_(Answer: App becomes unusable - 30-60 second load times!)_

---

### **Part 10: Best Practices to Follow**

**"As you modify this code, remember:**

**✅ Data Types**

- Use `.astype(float)` for hours, capacity, numeric values
- Use `.astype(int)` only for IDs, counts

**✅ Date Handling**

- Always parse with `dayfirst=True`
- Store as strings: `YYYY-MM-DD` format
- Display as `DD-MM-YYYY` for users

**✅ Caching**

- Check cache freshness before loading
- Clear cache after data modifications
- Return `.copy()` to prevent corruption

**✅ Error Handling**

- Initialize variables with safe defaults
- Use `errors='coerce'` in pd.to_datetime()
- Add try-except around data operations

**✅ Logging**

- Add diagnostic logs for debugging
- Use pattern: `[TAG] PSID {X}: field={value}`
- Helps trace issues in production

**✅ Testing**

- Test fractional hours (0.8h, 0.5h)
- Test edge cases (0h, over-allocation)
- Test date boundaries (month-end, year-end)

---

### **Part 11: How to Debug Production Issues**

**"When users report issues, follow this checklist:**

**1. Check Logs**

```
Look for: [ERROR], [WARNING], [DEBUG] tags
Find: Stack traces, variable values
```

**2. Clear Cache**

```
POST /api/clear_cache
Wait 30 seconds
Try again
```

**3. Verify Data Source**

```
Check Dataiku datasets:
- Is PSID present?
- Are dates in range?
- Is data type correct?
```

**4. Add Diagnostic Logs**

```python
logging.info(f"[DEBUG] PSID {psid}: available={X}, booked={Y}")
```

**5. Test with Known Good Data**

```
Pick a PSID/date you KNOW works
Compare with problematic one
Find the difference
```

**6. Check Recent Changes**

```
Did availability file update?
Did booking schema change?
Did column names change?
```

---

### **Part 12: Future Improvements**

**"Some ideas for making this better:**

**1. Move to Real Database**

- Replace bookings.csv with PostgreSQL/SQLite
- Add unique constraints
- Enable transactions
- Proper indexes for performance

**2. Real-time Updates**

- WebSocket for live calendar updates
- No need to refresh page

**3. Better Concurrency**

- Lock mechanism for booking writes
- Prevent race conditions

**4. Audit Logging**

- Track who changed what when
- Compliance requirement

**5. Automated Testing**

- Unit tests for business logic
- Integration tests for APIs
- Prevent regression bugs

---

### **Session Close: Q&A**

**"That's the complete walkthrough! Key takeaways:**

**1. Caching is critical** - 30-second TTL makes app usable
**2. Distributed hours** - Spread total hours across available days
**3. Float precision** - Always use float for hours, never int
**4. NET vs GROSS** - Watch for double-accounting after midnight sync
**5. Date consistency** - Standardize format everywhere
**6. Duplicate removal** - Auto-runs every 30 seconds
**7. Performance** - Module-level loading + dict lookups = fast

**Questions? Let's open the code and go through any parts again."**

---

### **Suggested Follow-up Activities**

After the walkthrough, have new developers:

1. **Add a simple API endpoint**
   - Example: `GET /api/booking_count/<psid>`
   - Tests their understanding of Flask + Pandas

2. **Fix a synthetic bug**
   - Change `.astype(float)` back to `.astype(int)`
   - Have them debug and fix it

3. **Add logging to a function**
   - Pick a function without logs
   - Add diagnostic logging
   - Test with live data

4. **Modify business logic**
   - Change color threshold (3h → 2h)
   - Update calendar colors accordingly

5. **Review a real bug fix**
   - Show git diff for NET vs GROSS fix
   - Explain why each line changed

---

## Version History

### Latest Fixes (Feb 2026)

1. ✅ NET vs GROSS availability detection
2. ✅ Duplicate booking removal
3. ✅ Fractional hour support (0.8h)
4. ✅ Date format standardization in exports
5. ✅ Dynamic column name detection
6. ✅ UnboundLocalError fixes
7. ✅ Cache clearing endpoint

---

**Document Prepared For**: Knowledge Transfer Session  
**Audience**: Fresher Python Developer  
**Recommend**: Walk through each section with actual code examples during KT
