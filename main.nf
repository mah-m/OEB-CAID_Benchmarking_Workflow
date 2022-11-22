#!/usr/bin/env nextflow

if (params.help) {
	
	    log.info"""
	    ==============================================
		CAID BENCHMARKING PIPELINE 
		This pipeline was developed based of the TCGA CANCER DRIVER GENES BENCHMARKING PIPELINE 
	    ==============================================
	    Usage:
	    Run the pipeline with default parameters:
	    nextflow run main.nf -profile docker

	    Run with user parameters:

 	    nextflow run main.nf -profile docker --input {driver.genes.file} --public_ref_dir {validation.reference.file} --participant_id {tool.name} --goldstandard_dir {gold.standards.dir} --challenges {analyzed.cancer.types} --assess_dir {benchmark.data.dir} --augmented_assess_dir {benchmark.augmented_data.dir} --results_dir {output.dir}

	    Mandatory arguments:
                --input                 List of cancer genes prediction
                --community_id          Name or OEB permanent ID for the benchmarking community
                --public_ref_dir        Directory of the refrence files that has the input data is validated agains. For the CAID challenge, these file are the same as gold standards. 
                --participant_id        Name of the tool used for prediction
                --goldstandard_dir      Dir that contains metrics reference datasets 
                --challenges_ids        List of challenges selected by the user, separated by spaces
                --assess_dir            Dir where the input data for the benchmark are stored

	    Other options:
                --validation_result     The output directory where the results from validation step will be saved
                --augmented_assess_dir  Dir where the augmented data for the benchmark are stored
                --assessment_results    The output directory where the results from the computed metrics step will be saved
                --outdir                The output directory where the consolidation of the benchmark will be saved
                --statsdir              The output directory with nextflow statistics
                --data_model_export_dir The output dir where json file with benchmarking data model contents will be saved
                --otherdir              The output directory where custom results will be saved (no directory inside)
	    Flags:
                --help                  Display this message
	    """

	exit 1
} else {

	log.info """\
         ==============================================
         CAID BENCHMARKING PIPELINE
         ==============================================
         input file: ${params.input}
         benchmarking community = ${params.community_id}
         public reference directory : ${params.public_ref_dir}
         tool name : ${params.participant_id}
         metrics reference datasets: ${params.goldstandard_dir}
         selected cancer types: ${params.challenges_ids}
         benchmark data: ${params.assess_dir}
         augmented benchmark data: ${params.augmented_assess_dir}
         validation results directory: ${params.validation_result}
         assessment results directory: ${params.assessment_results}
         consolidated benchmark results directory: ${params.outdir}
         Statistics results about nextflow run: ${params.statsdir}
         Benchmarking data model file location: ${params.data_model_export_dir}
         Directory with community-specific results: ${params.otherdir}
         """

}


// input files

input_file = file(params.input)
ref_dir = Channel.fromPath( params.public_ref_dir, type: 'dir' )
tool_name = params.participant_id.replaceAll("\\s","_")
gold_standards_dir = Channel.fromPath(params.goldstandard_dir, type: 'dir' ) 
challenges = params.challenges_ids
benchmark_data = Channel.fromPath(params.assess_dir, type: 'dir' )
community_id = params.community_id

// output 
validation_file = file(params.validation_result)
assessment_file = file(params.assessment_results)
aggregation_dir = file(params.outdir, type: 'dir')
augmented_benchmark_data = file(params.augmented_assess_dir, type: 'dir')
// It is really a file in this implementation
data_model_export_dir = file(params.data_model_export_dir)
other_dir = file(params.otherdir, type: 'dir')


process validation {

	// validExitStatus 0,1
	tag "Validating input file format"
    // container "validation_image"
	
	publishDir "${validation_file.parent}", saveAs: { filename -> validation_file.name }, mode: 'copy'

	input:
	file input_file
	path ref_dir
	val challenges
	val tool_name
	val community_id

	output:
	val task.exitStatus into EXIT_STAT
	file 'validation.json' into validation_out
	
	"""
	python3 /app/validation.py -i $input_file -r $ref_dir -com $community_id -c $challenges -p $tool_name -o validation.json
	"""

}

process compute_metrics {

	tag "Computing benchmark metrics for submitted data"
	
	publishDir "${assessment_file.parent}", saveAs: { filename -> assessment_file.name }, mode: 'copy'

	input:
	val file_validated from EXIT_STAT
	file input_file
	val challenges
	path gold_standards_dir
	val tool_name
	val community_id

	output:
	file 'assessment.json' into assessment_out

	when:
	file_validated == 0

	"""
	python /app/compute_metrics.py -i $input_file -c $challenges -m $gold_standards_dir -p $tool_name -com $community_id -o assessment.json
	"""

}

process benchmark_consolidation {

	tag "Performing benchmark assessment and building plots"
	publishDir "${aggregation_dir.parent}", pattern: "aggregation_dir", saveAs: { filename -> aggregation_dir.name }, mode: 'copy'
	publishDir "${data_model_export_dir.parent}", pattern: "data_model_export.json", saveAs: { filename -> data_model_export_dir.name }, mode: 'copy'
	publishDir "${augmented_benchmark_data.parent}", pattern: "augmented_benchmark_data", saveAs: { filename -> augmented_benchmark_data.name }, mode: 'copy'

	input:
	path benchmark_data
	file assessment_out
	file validation_out
	
	output:
	path 'aggregation_dir', type: 'dir'
	path 'augmented_benchmark_data', type: 'dir'
	path 'data_model_export.json'

	"""
	cp -Lpr $benchmark_data augmented_benchmark_data
	python /app/manage_assessment_data.py -b augmented_benchmark_data -p $assessment_out -o aggregation_dir
	python /app/merge_data_model_files.py -p $validation_out -m $assessment_out -a aggregation_dir -o data_model_export.json
	"""

}


workflow.onComplete { 
	println ( workflow.success ? "Done!" : "Oops .. something went wrong" )
}
