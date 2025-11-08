# Incident Post-Mortem Template

## Incident Summary

**Incident ID**: [AUTO-GENERATED]
**Date/Time**: [START TIME] - [END TIME]
**Duration**: [TOTAL DURATION]
**Severity**: [CRITICAL/HIGH/MEDIUM/LOW]

### What Happened?
[Brief description of the incident from user perspective]

### Impact Assessment
- **Financial Impact**: [Dollar amount lost/gained, positions affected]
- **Operational Impact**: [Trading downtime, missed opportunities]
- **User Experience**: [Any user-facing issues]
- **System Resources**: [CPU, memory, network affected]

## Timeline

All times in UTC.

| Time | Event | Component | Action Taken |
|------|-------|-----------|--------------|
| 00:00 | Incident start | [Component] | [Detection method] |
| 00:05 | Alert triggered | [Alerting system] | [Initial response] |
| 00:10 | Investigation began | [Team member] | [Diagnostic steps] |
| 00:20 | Root cause identified | [Component] | [What was found] |
| 00:30 | Mitigation implemented | [Team member] | [Temporary fix] |
| 00:45 | Permanent fix deployed | [Team member] | [Code/config change] |
| 01:00 | Service restored | [Component] | [Verification steps] |

## Root Cause Analysis

### What Caused the Incident?

**Primary Cause**: [Detailed explanation]

**Contributing Factors**:
1. [Factor 1] - [Explanation]
2. [Factor 2] - [Explanation]
3. [Factor 3] - [Explanation]

### Why Did It Happen?

**Technical Details**:
- [Code issue, configuration problem, external dependency failure]
- [Stack trace, error logs, monitoring data]

**Process Issues**:
- [Missing monitoring, inadequate testing, poor documentation]
- [Communication breakdown, insufficient training]

**Environmental Factors**:
- [High market volatility, network congestion, hardware issues]
- [Scheduled maintenance, deployment conflicts]

## Resilience System Performance

### What Worked Well
- [Circuit breaker prevented cascade failures]
- [State recovery maintained position integrity]
- [Automatic restart restored service quickly]
- [Alerting notified team promptly]

### What Failed
- [Circuit breaker didn't open when it should have]
- [State recovery corrupted data]
- [Watchdog failed to restart service]
- [Alerting system overwhelmed with notifications]

### Resilience Metrics During Incident
- **Circuit Breaker Trips**: [Number]
- **Retry Attempts**: [Number successful/failed]
- **State Recovery Time**: [Seconds]
- **Data Loss**: [Amount of lost data/transactions]
- **False Positive Alerts**: [Number]

## Response Analysis

### Detection & Alerting
- **Detection Method**: [How was incident detected?]
- **Alert Effectiveness**: [Did alerts work? Were they actionable?]
- **Time to Detection**: [How long from incident start to alert?]

### Response Time
- **Time to Investigation**: [From alert to investigation start]
- **Time to Diagnosis**: [From investigation to root cause]
- **Time to Mitigation**: [From diagnosis to temporary fix]
- **Time to Resolution**: [From mitigation to permanent fix]

### Communication
- **Internal Communication**: [How well did team communicate?]
- **External Communication**: [Were stakeholders informed?]
- **Documentation**: [Was incident documented in real-time?]

## Recovery Process

### Immediate Actions Taken
1. [Action 1] - [Time taken, effectiveness]
2. [Action 2] - [Time taken, effectiveness]
3. [Action 3] - [Time taken, effectiveness]

### Verification Steps
1. [Check 1] - [Result, time taken]
2. [Check 2] - [Result, time taken]
3. [Check 3] - [Result, time taken]

### Rollback Plan
- [What was the rollback strategy?]
- [Was rollback needed? Why/why not?]

## Black Box Analysis

### Key Events Leading to Incident
```
[Extract from black box recorder]
- Event 1: [Timestamp] [Component] [Message]
- Event 2: [Timestamp] [Component] [Message]
- Event 3: [Timestamp] [Component] [Message]
```

### System State at Incident Time
```
CPU Usage: [Percentage]
Memory Usage: [Percentage]
Network Connections: [Count]
Open Positions: [Count]
Active Orders: [Count]
Circuit Breaker Status: [State]
Degradation Level: [Level]
```

### Performance Metrics
- **API Latency**: [Average during incident]
- **Error Rate**: [Percentage during incident]
- **Recovery Time**: [Time to full functionality]

## Lessons Learned

### What Went Well
1. [Positive aspect] - [Why it worked, how to maintain]
2. [Positive aspect] - [Why it worked, how to maintain]
3. [Positive aspect] - [Why it worked, how to maintain]

### What Could Be Improved
1. [Issue] - [Impact, proposed solution]
2. [Issue] - [Impact, proposed solution]
3. [Issue] - [Impact, proposed solution]

### Systemic Issues Identified
1. [System issue] - [Root cause, long-term fix]
2. [System issue] - [Root cause, long-term fix]
3. [System issue] - [Root cause, long-term fix]

## Action Items

### Immediate (Next 24 hours)
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]

### Short-term (Next week)
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]

### Long-term (Next month)
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]
- [ ] [Action item] - [Owner] - [Due date]

## Prevention Measures

### Technical Fixes
1. **Code Changes**:
   - [Specific code change needed]
   - [Testing requirements]
   - [Deployment plan]

2. **Configuration Updates**:
   - [Config parameter to change]
   - [New monitoring to add]
   - [Alert threshold to adjust]

3. **Infrastructure Improvements**:
   - [Hardware/software upgrade needed]
   - [Network architecture change]
   - [Backup system enhancement]

### Process Improvements
1. **Monitoring Enhancements**:
   - [New metric to track]
   - [Alert rule to create]
   - [Dashboard to build]

2. **Testing Improvements**:
   - [New test case to add]
   - [Test automation to implement]
   - [Chaos engineering exercise]

3. **Documentation Updates**:
   - [Runbook section to add]
   - [Procedure to document]
   - [Knowledge base article]

## Risk Assessment

### Likelihood of Recurrence
- **Before Fixes**: [HIGH/MEDIUM/LOW]
- **After Fixes**: [HIGH/MEDIUM/LOW]

### Impact if Recurs
- **Financial**: [Estimated dollar impact]
- **Operational**: [System downtime, user impact]
- **Reputational**: [Brand/trust damage]

### Risk Mitigation Score
```
Current Risk Level: [1-10 scale]
Target Risk Level: [1-10 scale]
Risk Reduction Achieved: [Percentage]
```

## Follow-up Review

### Review Date: [Date, 1 week after incident]
**Attendees**: [Team members]
**Actions Reviewed**: [Which action items completed]
**Effectiveness Assessment**: [Did fixes work as expected]

### Review Date: [Date, 1 month after incident]
**Attendees**: [Team members]
**Recurrence Check**: [Has incident recurred?]
**Process Effectiveness**: [Did new procedures work?]

## Incident Classification

**Category**: [Infrastructure/Application/Network/Human Error]
**Subcategory**: [Specific type within category]
**Detection Method**: [Alert/Monitoring/User Report/Automated Test]
**Resolution Method**: [Code Fix/Config Change/Process Change/Infrastructure]

## Supporting Data

### Logs Extract
```
[Paste relevant log entries here]
```

### Monitoring Screenshots
[Attach relevant Grafana dashboards, metrics charts]

### Code Changes
[Link to pull requests, commits that fixed the issue]

---

## Approval

**Incident Commander**: ____________________ Date: ________
**Technical Lead**: ____________________ Date: ________
**Business Owner**: ____________________ Date: ________

## Distribution List

- [ ] Incident Response Team
- [ ] Development Team
- [ ] Operations Team
- [ ] Business Stakeholders
- [ ] External Partners (if applicable)

---

*This template ensures comprehensive analysis of incidents to prevent recurrence and improve system reliability.*
