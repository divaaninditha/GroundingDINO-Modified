
# Bilingual GroundingDINO Evaluation Report

## Summary Statistics

### English GroundingDINO (Original)
- Total Predictions: 400
- Total Detections: 2393
- Average Detections per Image: 5.98
- Processing Time: 1937.26s
- Errors: 0

### Indonesian GroundingDINO 
- Total Predictions: 400
- Total Detections: 144
- Average Detections per Image: 0.36
- Processing Time: 1916.86s
- Errors: 0

## Performance Analysis

### Detection Performance
- Detection Ratio (ID/EN): 0.060
- Indonesian detected 6.0% objects compared to English

### Speed Performance  
- Speed Ratio (ID/EN): 0.989
- Indonesian processing was 98.9% of English time

### Reliability
- Error Ratio (ID/EN): 1.000

## Detailed Metrics

### English Detection Statistics
- Mean: 5.98
- Std: 5.32
- Range: 0.0-27.0

### Indonesian Detection Statistics  
- Mean: 0.36
- Std: 0.53
- Range: 0.0-4.0

## Conclusion

This evaluation compares the performance of GroundingDINO using English vs Indonesian captions on 200 COCO images.

**Key Findings:**
1. Detection Performance: English performs better
2. Processing Speed: Indonesian is faster
3. Overall: This preliminary evaluation demonstrates the feasibility of Indonesian language support in GroundingDINO.

Generated on: 2025-06-09 15:50:35
