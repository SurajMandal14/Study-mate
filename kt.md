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
