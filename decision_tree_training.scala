import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, IndexToString,
VectorIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import java.io.FileOutputStream
import scala.Console
import java.util.Properties
sqlContext.setConf("spark.sql.shuffle.partitions", "200")
val props = new Properties();
props.setProperty("XX:MaxPermSize", "1024");
props.put("XX:PermSize", "512m");
sqlContext.setConf(props)
def delete_hdfs_file(file_path:String):Unit = {
@transient val hadoopconf = new org.apache.hadoop.conf.Configuration();
val fsys = org.apache.hadoop.fs.FileSystem.get(hadoopconf);
org.apache.hadoop.fs.FileUtil.fullyDelete(fsys, new
org.apache.hadoop.fs.Path(file_path))
}
def copy_file_to_hdfs(in_file:String, out_File:String):Unit = {
@transient val hadoopconf = new org.apache.hadoop.conf.Configuration();
val fsys = org.apache.hadoop.fs.FileSystem.get(hadoopconf);
org.apache.hadoop.fs.FileUtil.copy(new java.io.File(in_file), fsys, new
org.apache.hadoop.fs.Path(out_File), false, hadoopconf)
}
/*remove pre-existing csv files -- optional!*/
delete_hdfs_file(hdfs_df_path);
delete_hdfs_file(hdfs_df_train_path);
delete_hdfs_file(hdfs_df_train_numeric_nasarezeroes_path);
delete_hdfs_file(hdfs_df_train_categoric_path);
delete_hdfs_file(hdfs_df_validate_path);
delete_hdfs_file(hdfs_df_test_path);
delete_hdfs_file(hdfs_df_test_numeric_nasarezeroes_path);
delete_hdfs_file(hdfs_df_test_categoric_path);
delete_hdfs_file(hdfs_df_test_catFeatures_path);
delete_hdfs_file(hdfs_df_train_catCode_path);
/*paths to csv files*/
val df_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation.csv";
val hdfs_df_path = "imad.khoury/merged_dfs_ordered_646features_reputation.csv"
copy_file_to_hdfs(df_path, hdfs_df_path);
val df_train_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_training.csv";
val hdfs_df_train_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_training.csv"
copy_file_to_hdfs(df_train_path, hdfs_df_train_path);
val df_train_numeric_nasarezeroes_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_training_numeric_
nasarezeroes.csv";
val hdfs_df_train_numeric_nasarezeroes_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_training_numeric_nasarezeroes
.csv"
copy_file_to_hdfs(df_train_numeric_nasarezeroes_path,
hdfs_df_train_numeric_nasarezeroes_path);
val df_train_categoric_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_training_categori
c.csv";
val hdfs_df_train_categoric_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_training_categoric.csv"
copy_file_to_hdfs(df_train_categoric_path, hdfs_df_train_categoric_path);
val df_validate_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_validation.csv";
val hdfs_df_validate_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_validation.csv"
copy_file_to_hdfs(df_validate_path, hdfs_df_validate_path);
val df_test_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_testing.csv";
val hdfs_df_test_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_testing.csv"
copy_file_to_hdfs(df_test_path, hdfs_df_test_path);
val df_test_numeric_nasarezeroes_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_testing_numeric_n
asarezeroes.csv";
val hdfs_df_test_numeric_nasarezeroes_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_testing_numeric_nasarezeroes.
csv"
copy_file_to_hdfs(df_test_numeric_nasarezeroes_path,
hdfs_df_test_numeric_nasarezeroes_path);
val df_test_categoric_path =
"/home/urlrs/imad.khoury/merged_dfs_ordered_646features_reputation_testing_categoric
.csv";
val hdfs_df_test_categoric_path =
"imad.khoury/merged_dfs_ordered_646features_reputation_testing_categoric.csv"
copy_file_to_hdfs(df_test_categoric_path, hdfs_df_test_categoric_path);
val df_train_catFeatures_path = "/home/urlrs/imad.khoury/catFeaturesMap.txt";
val hdfs_df_train_catFeatures_path = "imad.khoury/catFeaturesMap.txt"
copy_file_to_hdfs(df_train_catFeatures_path, hdfs_df_train_catFeatures_path);
val df_train_catCode_path = "/home/urlrs/imad.khoury/catCode_train.txt";
val hdfs_df_train_catCode_path = "imad.khoury/catCode_train.txt"
copy_file_to_hdfs(df_train_catCode_path, hdfs_df_train_catCode_path);
val df_test_catCode_path = "/home/urlrs/imad.khoury/catCode_test.txt";
val hdfs_df_test_catCode_path = "imad.khoury/catCode_test.txt"
copy_file_to_hdfs(df_test_catCode_path, hdfs_df_test_catCode_path);
/*load training set into hdfs*/
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_train = sqlContext.read.format("com.databricks.spark.csv");
df_train.option("header", "true");
df_train.option("inferSchema","true");
//df_train.option("nullValue", "NA");
val df_train_loaded = df_train.load(hdfs_df_train_path);
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_train_categoric = sqlContext.read.format("com.databricks.spark.csv");
df_train_categoric.option("header", "true");
df_train_categoric.option("inferSchema","true");
//df_train_categoric.option("nullValue", "NA");
val df_train_categoric_loaded =
df_train_categoric.load(hdfs_df_train_categoric_path);
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_train_numeric_nasarezeroes =
sqlContext.read.format("com.databricks.spark.csv");
df_train_numeric_nasarezeroes.option("header", "true");
df_train_numeric_nasarezeroes.option("inferSchema","true");
//df_train_numeric_nasarezeroes.option("nullValue", "NA");
val df_train_numeric_nasarezeroes_loaded =
df_train_numeric_nasarezeroes.load(hdfs_df_train_numeric_nasarezeroes_path);
/*load test set into hdfs*/
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_test = sqlContext.read.format("com.databricks.spark.csv");
df_test.option("header", "true");
df_test.option("inferSchema","true");
//df_test.option("nullValue", "NA");
val df_test_loaded = df_test.load(hdfs_df_test_path);
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_test_categoric = sqlContext.read.format("com.databricks.spark.csv");
df_test_categoric.option("header", "true");
df_test_categoric.option("inferSchema","true");
//df_test_categoric.option("nullValue", "NA");
val df_test_categoric_loaded = df_test_categoric.load(hdfs_df_test_categoric_path);
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_test_numeric_nasarezeroes =
sqlContext.read.format("com.databricks.spark.csv");
df_test_numeric_nasarezeroes.option("header", "true");
df_test_numeric_nasarezeroes.option("inferSchema","true");
//df_test_numeric_nasarezeroes.option("nullValue", "NA");
val df_test_numeric_nasarezeroes_loaded =
df_test_numeric_nasarezeroes.load(hdfs_df_test_numeric_nasarezeroes_path);
/*load validation set into hdfs*/
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df_validate = sqlContext.read.format("com.databricks.spark.csv");
df_validate.option("header", "true");
df_validate.option("inferSchema","true");
//df_validate.option("nullValue", "NA");
val df_validate_loaded = df_validate.load(hdfs_df_validate_path);
/*load full set into hdfs*/
val sqlContext = new org.apache.spark.sql.SQLContext(sc);
val df = sqlContext.read.format("com.databricks.spark.csv");
df.option("header", "true");
df.option("inferSchema","true");
//df.option("nullValue", "NA");
val df_loaded = df.load(hdfs_df_path);
/*manual repartitions*/
//val df_loaded_rep = df_train_loaded.repartition(50)
//val df_validate_loaded_rep = df_validate_loaded.repartition(50)
val df_test_loaded_rep = df_test_loaded.repartition(50)
val df_test_numeric_nasarezeroes_loaded_rep =
df_test_numeric_nasarezeroes_loaded.repartition(50)
val df_test_categoric_loaded_rep = df_test_categoric_loaded.repartition(50)
val df_train_numeric_nasarezeroes_loaded_rep =
df_train_numeric_nasarezeroes_loaded.repartition(50)
val df_train_loaded_rep = df_train_loaded.repartition(50)
val df_train_categoric_loaded_rep = df_train_categoric_loaded.repartition(50)
/*categories pre-model-pre-processing*/
//use scala reflection:
//val categoricalFields =
scala.io.Source.fromFile(hdfs_df_train_catFeatures_path).mkString;
//val categoricalFields =
sc.textFile(hdfs_df_train_catFeatures_path).toString.toMap;
//:
//:load /home/urlrs/imad.khoury/catFeaturesMap.txt
//val categoricalFeaturesInfo = categoricalFields;
/*learn decision tree model on training set*/
/*labels pre-processing*/
val labelIndexer = new
StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df_train_loade
d_rep)
//join numeric and categoric dataframes and drop resulting duplicate columns and id
column and empty column
val df_train_loaded_rep =
df_train_numeric_nasarezeroes_loaded_rep.join(df_train_categoric_loaded_rep).drop(df
_train_categoric_loaded_rep.col("label")).drop(df_train_categoric_loaded_rep.col("id
")).drop(df_train_categoric_loaded_rep.col("")).repartition(50)
val df_test_loaded_rep =
df_test_numeric_nasarezeroes_loaded_rep.join(df_test_categoric_loaded_rep).drop(df_t
est_categoric_loaded_rep.col("label")).drop(df_test_categoric_loaded_rep.col("id")).
drop(df_test_categoric_loaded_rep.col("")).repartition(50)
val fos = new FileOutputStream("imad.khoury/dataframe_train_schema.txt")
Console.setOut(fos)
df_train_loaded_rep.printSchema()
fos.close()
val fos = new FileOutputStream("imad.khoury/dataframe_test_schema.txt")
Console.setOut(fos)
df_test_loaded_rep.printSchema()
fos.close()
//val df_train_loaded_rep_cached = df_train_loaded_rep.cache()
val fos = new FileOutputStream("imad.khoury/testtrain.txt")
Console.setOut(fos)
df_train_loaded_rep_cached.select("etns_33_family").show(250)
fos.close()
//val df_test_loaded_rep_cached = df_test_loaded_rep.cache()
val fos = new FileOutputStream("imad.khoury/testtest.txt")
Console.setOut(fos)
df_test_loaded_rep_cached.select("etns_33_family").show(250)
fos.close()
/*features pre-processing -- training dataset*/
:
:load /home/urlrs/imad.khoury/catCode_train.txt
val fos = new FileOutputStream("imad.khoury/encoded_schema.txt")
Console.setOut(fos)
encoded.printSchema()
fos.close()
val assembled = assembler.transform(encoded)
//val assembled_rep = assembler.repartition(50)
/*decision tree model parameters*/
val numClasses = 2;
val impurity = "entropy";
val maxDepth = 20;
val maxBins = 185; ////spark suggestion to minimum 182 because one feature has 182
factors -- default 100
/*train decision tree*/
val dt = new
DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures
").setImpurity(impurity).setMaxBins(maxBins).setMaxDepth(maxDepth)
// Convert indexed labels back to original labels
val labelConverter = new
IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(l
abelIndexer.labels)
// Set ml pipeline and execute it
val pipeline = new Pipeline().setStages(Array(labelIndexer, dt))
val model = pipeline.fit(assembled)
//val model = pipeline.fit(assembled_rep)
/*features pre-processing -- testing dataset*/
//free some memory -- GC debug
val indexed = null;
val encoded = null;
val df_train_loaded_rep = null;
:
:load /home/urlrs/imad.khoury/catCode_test.txt
val fos = new FileOutputStream("imad.khoury/testIndexed.txt")
Console.setOut(fos)
assembled.select("etns_33_family").show(250)
fos.close()
val assembled = assembler.transform(encoded)
//val assembled_rep = assembler.repartition(50)
val predictions = model.transform(df_test_loaded_rep)
val fos = new FileOutputStream("imad.khoury/predictions_test.txt")
Console.setOut(fos)
predictions.select("predictedLabel", "label", "features").show(5)
fos.close()
// Select (prediction, true label) and compute test error
val evaluator = new
MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("pr
ediction").setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
