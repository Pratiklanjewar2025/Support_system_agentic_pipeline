# Payments Service High CPU Runbook

## Trigger
Start this procedure when average CPU utilization for the payments service remains above 85% for 10 minutes or when latency rises together with CPU saturation.

## Response procedure
1. Confirm whether traffic volume increased abnormally.
2. Compare CPU usage across replicas. One hot replica can indicate a stuck worker or uneven traffic distribution.
3. Check application latency, request rate, error rate, and queue depth before scaling.
4. Inspect recent deployments and feature-flag changes.
5. Review logs for retry storms, timeout loops, serialization errors, or runaway background jobs.

## Scaling actions
- **Horizontal scale:** `kubectl scale deployment payments-service --replicas=<N> -n payments`
- **Verify HPA limits** are not preventing autoscaling: `kubectl get hpa -n payments`
- Do not scale beyond 2x current replicas in a single step without DB capacity confirmation.

## Root cause patterns
| Symptom | Likely Cause |
|---|---|
| One hot pod, others normal | Sticky sessions or stuck goroutine/thread |
| All pods elevated, spike in traffic | Legitimate load — scale out |
| All pods elevated, normal traffic | CPU-heavy regression in latest deploy |
| Elevated after batch job | Batch job not rate-limited — add throttle |

## Mitigation without scaling
- Enable request rate limiting at the ingress layer.
- Temporarily disable non-critical background jobs.
- Shed low-priority traffic via feature flags.

## Escalation
If CPU does not drop below 80% within 20 minutes of mitigation, escalate to the payments engineering team and open a P1 incident.
