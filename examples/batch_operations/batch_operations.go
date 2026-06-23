// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

// Batch Operations Example
//
// This example demonstrates efficient bulk data operations with LanceDB
// using the Go SDK. It covers:
// - Bulk data insertion strategies
// - Batch update patterns
// - Efficient bulk deletion
// - Memory management for large datasets
// - Performance optimization techniques
// - Error handling in batch operations

package main

import (
	"context"
	"fmt"
	. "github.com/eozsahin1993/lancedb-go/pkg/contracts"
	"github.com/eozsahin1993/lancedb-go/pkg/lancedb"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
)

const (
	BatchSize        = 1000  // Records per batch
	TotalRecords     = 10000 // Total records to process
	VectorDimensions = 256   // Embedding dimensions
)

type BatchRecord struct {
	ID          int32
	Name        string
	Description string
	Category    string
	Value       float64
	Vector      []float32
}

func main() {
	fmt.Println("📦 LanceDB Go SDK - Batch Operations Example")
	fmt.Println("==================================================")

	// Setup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tempDir, err := os.MkdirTemp("", "lancedb_batch_example_")
	if err != nil {
		log.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	fmt.Printf("📂 Using database directory: %s\n", tempDir)

	// Connect to database
	fmt.Println("\n📋 Step 1: Setting up database for batch operations...")
	conn, err := lancedb.Connect(ctx, tempDir, nil)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	// Create table optimized for batch operations
	table, schema, err := createBatchTable(conn, ctx)
	if err != nil {
		log.Fatalf("Failed to create batch table: %v", err)
	}
	defer table.Close()
	fmt.Printf("✅ Created table optimized for batch operations\n")

	// Demonstrate different batch insertion strategies
	fmt.Println("\n📋 Step 2: Batch insertion strategies...")
	if err := demonstrateBatchInsert(table, schema); err != nil {
		log.Fatalf("Failed batch insert demo: %v", err)
	}

	// Demonstrate batch update operations
	fmt.Println("\n📋 Step 3: Batch update operations...")
	if err := demonstrateBatchUpdate(table); err != nil {
		log.Fatalf("Failed batch update demo: %v", err)
	}

	// Demonstrate memory-efficient processing
	fmt.Println("\n📋 Step 4: Memory-efficient large dataset processing...")
	if err := demonstrateMemoryEfficientProcessing(table, schema); err != nil {
		log.Fatalf("Failed memory-efficient processing: %v", err)
	}

	// Demonstrate concurrent batch operations
	fmt.Println("\n📋 Step 5: Concurrent batch operations...")
	if err := demonstrateConcurrentOperations(conn, ctx); err != nil {
		log.Fatalf("Failed concurrent operations: %v", err)
	}

	// Demonstrate batch deletion strategies
	fmt.Println("\n📋 Step 6: Batch deletion strategies...")
	if err := demonstrateBatchDeletion(table); err != nil {
		log.Fatalf("Failed batch deletion demo: %v", err)
	}

	// Performance analysis
	fmt.Println("\n📋 Step 7: Performance analysis and optimization...")
	if err := performanceAnalysis(table, schema); err != nil {
		log.Fatalf("Failed performance analysis: %v", err)
	}

	// Error handling and recovery
	fmt.Println("\n📋 Step 8: Error handling and recovery patterns...")
	if err := errorHandlingPatterns(table, schema); err != nil {
		log.Fatalf("Failed error handling demo: %v", err)
	}

	fmt.Println("\n🎉 Batch operations examples completed successfully!")
	fmt.Println("==================================================")
}

func createBatchTable(conn IConnection, ctx context.Context) (ITable, *arrow.Schema, error) {
	fields := []arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "description", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "category", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "value", Type: arrow.PrimitiveTypes.Float64, Nullable: false},
		{Name: "vector", Type: arrow.FixedSizeListOf(VectorDimensions, arrow.PrimitiveTypes.Float32), Nullable: false},
	}

	arrowSchema := arrow.NewSchema(fields, nil)
	schema, err := lancedb.NewSchema(arrowSchema)
	if err != nil {
		return nil, nil, err
	}

	table, err := conn.CreateTable(ctx, "batch_data", schema)
	if err != nil {
		return nil, nil, err
	}
	return table, arrowSchema, nil
}

func demonstrateBatchInsert(table ITable, schema *arrow.Schema) error {
	fmt.Println("  📥 Batch Insertion Strategies")

	// Strategy 1: Single large batch
	fmt.Println("  🔹 Strategy 1: Single large batch insertion")
	start := time.Now()

	largeDataset := generateBatchData(5000, 1)
	if err := insertBatch(table, schema, largeDataset); err != nil {
		return fmt.Errorf("large batch insert failed: %w", err)
	}

	largeBatchTime := time.Since(start)
	fmt.Printf("    ⏱️ Large batch (5000 records): %v\n", largeBatchTime)

	// Strategy 2: Multiple medium batches
	fmt.Println("\n  🔹 Strategy 2: Multiple medium batches")
	start = time.Now()

	batchSizes := []int{1000, 1000, 1000, 1000, 1000}
	startID := int32(5001)

	for i, batchSize := range batchSizes {
		batchData := generateBatchData(batchSize, startID)
		if err := insertBatch(table, schema, batchData); err != nil {
			return fmt.Errorf("medium batch %d failed: %w", i+1, err)
		}
		startID += int32(batchSize)
	}

	mediumBatchTime := time.Since(start)
	fmt.Printf("    ⏱️ Medium batches (5x 1000 records): %v\n", mediumBatchTime)

	// Strategy 3: Many small batches
	fmt.Println("\n  🔹 Strategy 3: Many small batches")
	start = time.Now()

	numSmallBatches := 50
	smallBatchSize := 100
	startID = 10001

	for i := 0; i < numSmallBatches; i++ {
		batchData := generateBatchData(smallBatchSize, startID)
		if err := insertBatch(table, schema, batchData); err != nil {
			return fmt.Errorf("small batch %d failed: %w", i+1, err)
		}
		startID += int32(smallBatchSize)
	}

	smallBatchTime := time.Since(start)
	fmt.Printf("    ⏱️ Small batches (50x 100 records): %v\n", smallBatchTime)

	// Performance comparison
	fmt.Println("\n  📊 Batch Size Performance Analysis:")
	fmt.Printf("    Large batch:  %.2f records/second\n", 5000.0/largeBatchTime.Seconds())
	fmt.Printf("    Medium batch: %.2f records/second\n", 5000.0/mediumBatchTime.Seconds())
	fmt.Printf("    Small batch:  %.2f records/second\n", 5000.0/smallBatchTime.Seconds())

	// Get final count
	count, err := table.Count(context.Background())
	if err != nil {
		return fmt.Errorf("failed to get count: %w", err)
	}
	fmt.Printf("\n  📊 Total records in table: %d\n", count)

	fmt.Println("\n  💡 Batch Size Recommendations:")
	fmt.Println("    • 1000-5000 records: Good balance of performance and memory usage")
	fmt.Println("    • Larger batches: Better throughput but higher memory usage")
	fmt.Println("    • Smaller batches: Better for memory-constrained environments")
	fmt.Println("    • Consider network latency and available memory when choosing size")

	return nil
}

func demonstrateBatchUpdate(table ITable) error {
	fmt.Println("  ✏️ Batch Update Operations")

	// Pattern 1: Category-based batch update
	fmt.Println("  🔹 Pattern 1: Category-based updates")
	categories := []string{"electronics", "books", "clothing", "home", "sports"}

	for i, category := range categories {
		newValue := float64(100 + i*50) // 100, 150, 200, 250, 300

		start := time.Now()
		updates := map[string]interface{}{
			"value": newValue,
		}

		predicate := fmt.Sprintf("category = '%s'", category)
		err := table.Update(context.Background(), predicate, updates)
		updateTime := time.Since(start)

		if err != nil {
			fmt.Printf("    ⚠️ Update for category '%s' failed: %v\n", category, err)
			continue
		}

		// Verify update
		results, err := table.SelectWithFilter(context.Background(), predicate)
		if err != nil {
			fmt.Printf("    ⚠️ Verification for category '%s' failed: %v\n", category, err)
			continue
		}

		fmt.Printf("    ✅ Updated %d '%s' records to value %.0f (%v)\n",
			len(results), category, newValue, updateTime)
	}

	// Pattern 2: Range-based batch update
	fmt.Println("\n  🔹 Pattern 2: Range-based updates")

	rangeUpdates := []struct {
		condition string
		newValue  float64
		desc      string
	}{
		{"value < 150", 125.0, "low-value items"},
		{"value BETWEEN 150 AND 250", 200.0, "medium-value items"},
		{"value > 250", 300.0, "high-value items"},
	}

	for _, update := range rangeUpdates {
		start := time.Now()
		updates := map[string]interface{}{
			"value": update.newValue,
		}

		err := table.Update(context.Background(), update.condition, updates)
		updateTime := time.Since(start)

		if err != nil {
			fmt.Printf("    ⚠️ Range update for %s failed: %v\n", update.desc, err)
			continue
		}

		fmt.Printf("    ✅ Updated %s to %.0f (%v)\n", update.desc, update.newValue, updateTime)
	}

	// Pattern 3: ID-based updates
	fmt.Println("\n  🔹 Pattern 3: ID-based batch updates")

	// Update records based on ID ranges

	start := time.Now()
	updates := map[string]interface{}{
		"value": 999.0, // Mark recent records with special value
	}

	// Update records with IDs in a specific range
	predicate := "id BETWEEN 1000 AND 2000"
	err := table.Update(context.Background(), predicate, updates)
	updateTime := time.Since(start)

	if err != nil {
		fmt.Printf("    ⚠️ ID-based update failed: %v\n", err)
	} else {
		fmt.Printf("    ✅ Updated records in ID range 1000-2000 (%v)\n", updateTime)
	}

	fmt.Println("\n  💡 Batch Update Best Practices:")
	fmt.Println("    • Use selective predicates to minimize affected records")
	fmt.Println("    • Consider indexing frequently updated columns")
	fmt.Println("    • Batch related updates together for better performance")
	fmt.Println("    • Monitor update performance and adjust batch sizes accordingly")

	return nil
}

func demonstrateMemoryEfficientProcessing(table ITable, schema *arrow.Schema) error {
	fmt.Println("  🧠 Memory-Efficient Large Dataset Processing")

	// Simulate processing a very large dataset in chunks
	fmt.Println("  🔹 Processing large dataset in memory-efficient chunks")

	totalRecordsToProcess := 25000
	chunkSize := 2000
	processedCount := 0

	var memStatsBefore, memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	start := time.Now()

	for startID := int32(20000); processedCount < totalRecordsToProcess; startID += int32(chunkSize) {
		currentChunkSize := chunkSize
		if processedCount+chunkSize > totalRecordsToProcess {
			currentChunkSize = totalRecordsToProcess - processedCount
		}

		// Generate chunk
		chunkData := generateBatchData(currentChunkSize, startID)

		// Process chunk (insert)
		if err := insertBatch(table, schema, chunkData); err != nil {
			return fmt.Errorf("chunk processing failed at record %d: %w", startID, err)
		}

		processedCount += currentChunkSize

		// Force garbage collection periodically to manage memory
		if processedCount%10000 == 0 {
			runtime.GC()
			fmt.Printf("    📊 Processed %d/%d records (%.1f%%)\n",
				processedCount, totalRecordsToProcess,
				float64(processedCount)/float64(totalRecordsToProcess)*100)
		}
	}

	processingTime := time.Since(start)
	runtime.ReadMemStats(&memStatsAfter)

	fmt.Printf("  ✅ Processed %d records in %v\n", totalRecordsToProcess, processingTime)
	fmt.Printf("  📊 Memory usage: %.2f MB allocated, %.2f MB in use\n",
		float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc)/1024/1024,
		float64(memStatsAfter.Alloc-memStatsBefore.Alloc)/1024/1024)

	// Demonstrate streaming query processing
	fmt.Println("\n  🔹 Streaming query results for large datasets")

	// Simulate processing query results in batches
	limit := 5000
	offset := 0
	totalProcessed := 0

	start = time.Now()

	for {
		results, err := table.SelectWithLimit(context.Background(), limit, offset)
		if err != nil {
			return fmt.Errorf("streaming query failed: %w", err)
		}

		if len(results) == 0 {
			break
		}

		// Process results batch (simulated)
		processResultsBatch(results)

		totalProcessed += len(results)
		offset += limit

		if len(results) < limit {
			break // Last batch
		}

		fmt.Printf("    📊 Processed %d results so far...\n", totalProcessed)
	}

	streamingTime := time.Since(start)
	fmt.Printf("  ✅ Streamed and processed %d results in %v\n", totalProcessed, streamingTime)

	fmt.Println("\n  💡 Memory-Efficient Processing Tips:")
	fmt.Println("    • Process data in fixed-size chunks to control memory usage")
	fmt.Println("    • Use streaming queries for large result sets")
	fmt.Println("    • Force garbage collection periodically in long-running processes")
	fmt.Println("    • Monitor memory usage and adjust chunk sizes accordingly")
	fmt.Println("    • Consider using worker pools for parallel chunk processing")

	return nil
}

func demonstrateConcurrentOperations(conn IConnection, ctx context.Context) error {
	fmt.Println("  🔄 Concurrent Batch Operations")

	// Create separate table for concurrent operations
	table, schema, err := createConcurrentTable(conn, ctx)
	if err != nil {
		return fmt.Errorf("failed to create concurrent table: %w", err)
	}
	defer table.Close()

	// Pattern 1: Concurrent inserts from multiple goroutines
	fmt.Println("  🔹 Concurrent batch inserts")

	numWorkers := 4
	recordsPerWorker := 2500

	var wg sync.WaitGroup
	var insertMutex sync.Mutex
	insertErrors := make([]error, 0)

	start := time.Now()

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			startID := int32(workerID*recordsPerWorker + 1)
			workerData := generateBatchData(recordsPerWorker, startID)

			if err := insertBatch(table, schema, workerData); err != nil {
				insertMutex.Lock()
				insertErrors = append(insertErrors, fmt.Errorf("worker %d: %w", workerID, err))
				insertMutex.Unlock()
				return
			}

			fmt.Printf("    ✅ Worker %d completed %d inserts\n", workerID, recordsPerWorker)
		}(i)
	}

	wg.Wait()
	concurrentInsertTime := time.Since(start)

	if len(insertErrors) > 0 {
		fmt.Printf("  ⚠️ %d workers encountered errors:\n", len(insertErrors))
		for _, err := range insertErrors {
			fmt.Printf("    • %v\n", err)
		}
	} else {
		fmt.Printf("  ✅ All %d workers completed successfully in %v\n", numWorkers, concurrentInsertTime)

		// Verify total count
		count, err := table.Count(context.Background())
		if err == nil {
			fmt.Printf("    📊 Total records inserted: %d\n", count)
		}
	}

	// Pattern 2: Producer-Consumer pattern
	fmt.Println("\n  🔹 Producer-Consumer batch processing")

	batchChannel := make(chan []BatchRecord, 5) // Buffer 5 batches
	var producerWg, consumerWg sync.WaitGroup

	// Producer goroutine
	producerWg.Add(1)
	go func() {
		defer producerWg.Done()
		defer close(batchChannel)

		batchID := int32(20000)
		for i := 0; i < 10; i++ { // Produce 10 batches
			batch := generateBatchData(500, batchID)
			batchChannel <- batch
			batchID += 500
			time.Sleep(100 * time.Millisecond) // Simulate production time
		}
		fmt.Println("    📤 Producer finished generating batches")
	}()

	// Consumer goroutines
	numConsumers := 2
	for i := 0; i < numConsumers; i++ {
		consumerWg.Add(1)
		go func(consumerID int) {
			defer consumerWg.Done()

			batchCount := 0
			for batch := range batchChannel {
				if err := insertBatch(table, schema, batch); err != nil {
					fmt.Printf("    ⚠️ Consumer %d failed batch %d: %v\n", consumerID, batchCount, err)
					continue
				}
				batchCount++
				fmt.Printf("    📥 Consumer %d processed batch %d (%d records)\n",
					consumerID, batchCount, len(batch))
			}
		}(i)
	}

	// Wait for completion
	producerWg.Wait()
	consumerWg.Wait()

	fmt.Println("    ✅ Producer-Consumer pattern completed")

	fmt.Println("\n  💡 Concurrent Operations Guidelines:")
	fmt.Println("    • Use connection pooling for concurrent database access")
	fmt.Println("    • Implement proper error handling and retry mechanisms")
	fmt.Println("    • Monitor resource usage (connections, memory, CPU)")
	fmt.Println("    • Use channels for coordinating between goroutines")
	fmt.Println("    • Consider batch size vs. concurrency level trade-offs")

	return nil
}

func demonstrateBatchDeletion(table ITable) error {
	fmt.Println("  🗑️ Batch Deletion Strategies")

	// Get initial count
	initialCount, err := table.Count(context.Background())
	if err != nil {
		return fmt.Errorf("failed to get initial count: %w", err)
	}
	fmt.Printf("  📊 Initial record count: %d\n", initialCount)

	// Strategy 1: Category-based deletion
	fmt.Println("\n  🔹 Strategy 1: Category-based batch deletion")

	categoriesToDelete := []string{"electronics", "books"}

	for _, category := range categoriesToDelete {
		start := time.Now()

		// First, count records to be deleted
		results, err := table.SelectWithFilter(context.Background(), fmt.Sprintf("category = '%s'", category))
		if err != nil {
			fmt.Printf("    ⚠️ Failed to count %s records: %v\n", category, err)
			continue
		}

		recordsToDelete := len(results)

		// Delete the records
		err = table.Delete(context.Background(), fmt.Sprintf("category = '%s'", category))
		deleteTime := time.Since(start)

		if err != nil {
			fmt.Printf("    ⚠️ Failed to delete %s records: %v\n", category, err)
			continue
		}

		fmt.Printf("    ✅ Deleted %d '%s' records (%v)\n", recordsToDelete, category, deleteTime)
	}

	// Strategy 2: Value-based range deletion
	fmt.Println("\n  🔹 Strategy 2: Range-based batch deletion")

	rangesToDelete := []struct {
		condition string
		desc      string
	}{
		{"value < 100", "low-value records"},
		{"value > 900", "high-value records"},
	}

	for _, rangeDelete := range rangesToDelete {
		start := time.Now()

		err := table.Delete(context.Background(), rangeDelete.condition)
		deleteTime := time.Since(start)

		if err != nil {
			fmt.Printf("    ⚠️ Failed to delete %s: %v\n", rangeDelete.desc, err)
			continue
		}

		fmt.Printf("    ✅ Deleted %s (%v)\n", rangeDelete.desc, deleteTime)
	}

	// Strategy 3: ID-based batch deletion (cleanup old records)
	fmt.Println("\n  🔹 Strategy 3: ID-based cleanup (old records)")

	start := time.Now()
	err = table.Delete(context.Background(), "id < 5000") // Delete first 5000 records
	deleteTime := time.Since(start)

	if err != nil {
		fmt.Printf("    ⚠️ Failed to delete old records: %v\n", err)
	} else {
		fmt.Printf("    ✅ Deleted old records (ID < 5000) in %v\n", deleteTime)
	}

	// Final count
	finalCount, err := table.Count(context.Background())
	if err != nil {
		return fmt.Errorf("failed to get final count: %w", err)
	}

	deletedCount := initialCount - finalCount
	fmt.Printf("\n  📊 Deletion Summary:\n")
	fmt.Printf("    Initial: %d records\n", initialCount)
	fmt.Printf("    Final:   %d records\n", finalCount)
	fmt.Printf("    Deleted: %d records\n", deletedCount)

	fmt.Println("\n  💡 Batch Deletion Best Practices:")
	fmt.Println("    • Use selective predicates to avoid unintended deletions")
	fmt.Println("    • Consider the impact on indexes and query performance")
	fmt.Println("    • Implement soft deletes for recoverable operations")
	fmt.Println("    • Monitor storage reclamation after large deletions")
	fmt.Println("    • Use transactions for critical deletion operations")

	return nil
}

func performanceAnalysis(table ITable, schema *arrow.Schema) error {
	fmt.Println("  ⚡ Performance Analysis and Optimization")

	// Test different batch sizes
	fmt.Println("  🔹 Batch size performance comparison")

	batchSizes := []int{100, 500, 1000, 2000, 5000}
	testRecords := 5000

	for _, batchSize := range batchSizes {
		// Clean slate for each test (using a subset of data)
		numBatches := testRecords / batchSize
		startID := int32(50000 + batchSize*10) // Unique ID range for each test

		start := time.Now()

		for i := 0; i < numBatches; i++ {
			currentBatchSize := batchSize
			if i == numBatches-1 && testRecords%batchSize != 0 {
				currentBatchSize = testRecords % batchSize
			}

			batchData := generateBatchData(currentBatchSize, startID+int32(i*batchSize))
			if err := insertBatch(table, schema, batchData); err != nil {
				return fmt.Errorf("performance test batch failed: %w", err)
			}
		}

		elapsed := time.Since(start)
		throughput := float64(testRecords) / elapsed.Seconds()

		fmt.Printf("    Batch size %d: %v (%.0f records/sec)\n", batchSize, elapsed, throughput)
	}

	// Memory usage analysis
	fmt.Println("\n  🔹 Memory usage analysis")

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	// Process a large batch
	largeBatch := generateBatchData(10000, 60000)
	runtime.ReadMemStats(&m2)

	memUsed := float64(m2.Alloc-m1.Alloc) / 1024 / 1024
	fmt.Printf("    Memory for 10K records: %.2f MB\n", memUsed)

	// Insert the batch
	start := time.Now()
	if err := insertBatch(table, schema, largeBatch); err != nil {
		return fmt.Errorf("large batch insert failed: %w", err)
	}
	insertTime := time.Since(start)

	fmt.Printf("    Insert time: %v (%.0f records/sec)\n",
		insertTime, 10000.0/insertTime.Seconds())

	// Query performance with different result sizes
	fmt.Println("\n  🔹 Query performance analysis")

	queryLimits := []int{100, 1000, 5000, 10000}

	for _, limit := range queryLimits {
		start := time.Now()
		results, err := table.SelectWithLimit(context.Background(), limit, 0)
		queryTime := time.Since(start)

		if err != nil {
			fmt.Printf("    Query limit %d failed: %v\n", limit, err)
			continue
		}

		fmt.Printf("    Query %d records: %v (%.0f records/sec)\n",
			len(results), queryTime, float64(len(results))/queryTime.Seconds())
	}

	fmt.Println("\n  📊 Performance Optimization Recommendations:")
	fmt.Println("    • Optimal batch size: 1000-2000 records for most use cases")
	fmt.Println("    • Monitor memory usage for large batches")
	fmt.Println("    • Use appropriate indexes for frequently queried columns")
	fmt.Println("    • Consider parallel processing for CPU-intensive operations")
	fmt.Println("    • Profile your specific workload to find optimal parameters")

	return nil
}

func errorHandlingPatterns(table ITable, schema *arrow.Schema) error {
	fmt.Println("  🛡️ Error Handling and Recovery Patterns")

	// Pattern 1: Retry mechanism for transient failures
	fmt.Println("  🔹 Retry mechanism for batch operations")

	maxRetries := 3
	retryDelay := 100 * time.Millisecond

	retryInsert := func(data []BatchRecord) error {
		for attempt := 1; attempt <= maxRetries; attempt++ {
			err := insertBatch(table, schema, data)
			if err == nil {
				return nil
			}

			fmt.Printf("    ⚠️ Attempt %d failed: %v\n", attempt, err)

			if attempt < maxRetries {
				time.Sleep(retryDelay * time.Duration(attempt))
			}
		}
		return fmt.Errorf("failed after %d attempts", maxRetries)
	}

	// Test retry mechanism
	testData := generateBatchData(500, 70000)
	if err := retryInsert(testData); err != nil {
		fmt.Printf("    ❌ Retry mechanism failed: %v\n", err)
	} else {
		fmt.Printf("    ✅ Retry mechanism succeeded\n")
	}

	// Pattern 2: Partial batch processing with error recovery
	fmt.Println("\n  🔹 Partial batch processing with error recovery")

	processWithRecovery := func(largeBatch []BatchRecord, subBatchSize int) error {
		successCount := 0
		failureCount := 0

		for i := 0; i < len(largeBatch); i += subBatchSize {
			end := i + subBatchSize
			if end > len(largeBatch) {
				end = len(largeBatch)
			}

			subBatch := largeBatch[i:end]

			if err := insertBatch(table, schema, subBatch); err != nil {
				fmt.Printf("    ⚠️ Sub-batch %d-%d failed: %v\n", i, end-1, err)
				failureCount++

				// Try individual records in failed batch
				for j, record := range subBatch {
					singleRecord := []BatchRecord{record}
					if err := insertBatch(table, schema, singleRecord); err != nil {
						fmt.Printf("    ❌ Record %d failed: %v\n", i+j, err)
					} else {
						successCount++
					}
				}
			} else {
				successCount += len(subBatch)
			}
		}

		fmt.Printf("    📊 Processing complete: %d success, %d failures\n",
			successCount, failureCount)
		return nil
	}

	// Test partial processing
	testBatch := generateBatchData(2000, 71000)
	if err := processWithRecovery(testBatch, 500); err != nil {
		return fmt.Errorf("partial processing failed: %w", err)
	}

	// Pattern 3: Validation before batch operations
	fmt.Println("\n  🔹 Data validation before batch operations")

	validateAndInsert := func(data []BatchRecord) error {
		// Pre-validation
		invalidRecords := 0
		for i, record := range data {
			if record.Name == "" || record.Value < 0 || len(record.Vector) != VectorDimensions {
				fmt.Printf("    ⚠️ Invalid record at index %d\n", i)
				invalidRecords++
			}
		}

		if invalidRecords > 0 {
			return fmt.Errorf("found %d invalid records, aborting batch", invalidRecords)
		}

		// If validation passes, proceed with insertion
		return insertBatch(table, schema, data)
	}

	// Test with valid data
	validData := generateBatchData(300, 73000)
	if err := validateAndInsert(validData); err != nil {
		fmt.Printf("    ❌ Valid data insertion failed: %v\n", err)
	} else {
		fmt.Printf("    ✅ Valid data inserted successfully\n")
	}

	// Test with some invalid data
	invalidData := generateBatchData(100, 73500)
	// Introduce some invalid records
	invalidData[10].Name = ""  // Invalid name
	invalidData[20].Value = -1 // Invalid value

	if err := validateAndInsert(invalidData); err != nil {
		fmt.Printf("    ✅ Invalid data correctly rejected: %v\n", err)
	}

	fmt.Println("\n  💡 Error Handling Best Practices:")
	fmt.Println("    • Implement exponential backoff for retry mechanisms")
	fmt.Println("    • Validate data before expensive operations")
	fmt.Println("    • Use partial processing to handle large batches gracefully")
	fmt.Println("    • Log detailed error information for debugging")
	fmt.Println("    • Implement circuit breakers for persistent failures")
	fmt.Println("    • Monitor error rates and success rates")

	return nil
}

// Helper functions

func generateBatchData(count int, startID int32) []BatchRecord {
	rand.Seed(time.Now().UnixNano() + int64(startID))

	categories := []string{"electronics", "books", "clothing", "home", "sports"}
	records := make([]BatchRecord, count)

	for i := 0; i < count; i++ {
		category := categories[rand.Intn(len(categories))]

		records[i] = BatchRecord{
			ID:          startID + int32(i),
			Name:        fmt.Sprintf("Item %d", startID+int32(i)),
			Description: fmt.Sprintf("Description for item %d in category %s", startID+int32(i), category),
			Category:    category,
			Value:       rand.Float64() * 1000,
			Vector:      generateRandomVector(VectorDimensions),
		}
	}

	return records
}

func generateRandomVector(dimensions int) []float32 {
	vector := make([]float32, dimensions)
	for i := 0; i < dimensions; i++ {
		vector[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}

	// Normalize vector
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm > 0 {
		for i := range vector {
			vector[i] /= norm
		}
	}

	return vector
}

func insertBatch(table ITable, schema *arrow.Schema, data []BatchRecord) error {
	if table == nil {
		return fmt.Errorf("table is nil")
	}
	if schema == nil {
		return fmt.Errorf("schema is nil")
	}

	pool := memory.NewGoAllocator()

	// Prepare arrays
	ids := make([]int32, len(data))
	names := make([]string, len(data))
	descriptions := make([]string, len(data))
	categories := make([]string, len(data))
	values := make([]float64, len(data))
	allVectors := make([]float32, len(data)*VectorDimensions)

	for i, record := range data {
		ids[i] = record.ID
		names[i] = record.Name
		descriptions[i] = record.Description
		categories[i] = record.Category
		values[i] = record.Value
		copy(allVectors[i*VectorDimensions:(i+1)*VectorDimensions], record.Vector)
	}

	// Build arrays
	idBuilder := array.NewInt32Builder(pool)
	idBuilder.AppendValues(ids, nil)
	idArray := idBuilder.NewArray()
	defer idArray.Release()

	nameBuilder := array.NewStringBuilder(pool)
	nameBuilder.AppendValues(names, nil)
	nameArray := nameBuilder.NewArray()
	defer nameArray.Release()

	descBuilder := array.NewStringBuilder(pool)
	descBuilder.AppendValues(descriptions, nil)
	descArray := descBuilder.NewArray()
	defer descArray.Release()

	catBuilder := array.NewStringBuilder(pool)
	catBuilder.AppendValues(categories, nil)
	catArray := catBuilder.NewArray()
	defer catArray.Release()

	valueBuilder := array.NewFloat64Builder(pool)
	valueBuilder.AppendValues(values, nil)
	valueArray := valueBuilder.NewArray()
	defer valueArray.Release()

	// Vector array
	vectorBuilder := array.NewFloat32Builder(pool)
	vectorBuilder.AppendValues(allVectors, nil)
	vectorFloat32Array := vectorBuilder.NewArray()
	defer vectorFloat32Array.Release()

	vectorListType := arrow.FixedSizeListOf(VectorDimensions, arrow.PrimitiveTypes.Float32)
	vectorArray := array.NewFixedSizeListData(
		array.NewData(vectorListType, len(data), []*memory.Buffer{nil},
			[]arrow.ArrayData{vectorFloat32Array.Data()}, 0, 0),
	)
	defer vectorArray.Release()

	// Create record
	columns := []arrow.Array{idArray, nameArray, descArray, catArray, valueArray, vectorArray}
	record := array.NewRecord(schema, columns, int64(len(data)))
	defer record.Release()

	return table.Add(context.Background(), record, nil)
}

func createConcurrentTable(conn IConnection, ctx context.Context) (ITable, *arrow.Schema, error) {
	fields := []arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "name", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "description", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "category", Type: arrow.BinaryTypes.String, Nullable: false},
		{Name: "value", Type: arrow.PrimitiveTypes.Float64, Nullable: false},
		{Name: "vector", Type: arrow.FixedSizeListOf(VectorDimensions, arrow.PrimitiveTypes.Float32), Nullable: false},
	}

	arrowSchema := arrow.NewSchema(fields, nil)
	schema, err := lancedb.NewSchema(arrowSchema)
	if err != nil {
		return nil, nil, err
	}

	table, err := conn.CreateTable(ctx, "concurrent_data", schema)
	return table, arrowSchema, nil
}

func processResultsBatch(results []map[string]interface{}) {
	// Simulate processing time
	time.Sleep(10 * time.Millisecond)

	// Could do actual processing here like:
	// - Data transformation
	// - Aggregations
	// - External API calls
	// - Writing to other systems
}
