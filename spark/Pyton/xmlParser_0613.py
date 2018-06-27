#pyspark --packages com.databricks:spark-xml_2.10:0.4.1,com.databricks:spark-csv_2.10:1.5.0
#pyspark --packages com.databricks:spark-xml_2.11:0.4.1,com.databricks:spark-csv_2.10:1.5.0

from pyspark.sql.functions import col,explode
from pyspark.sql.types import StructType,StructField
from pyspark.sql.types import *

inputDF = spark.read.format('com.databricks.spark.xml').options(rowTag='order').load('adl://yetiadls.azuredatalakestore.net/clusters/yetidpe3600/orders.xml')

#flattenedProductDF = inputDF.withColumn('product_lineitem', explode('product-lineitems.product-lineitem'))
#flattenedPaymentDF = flattenedProductDF.withColumn('payment_custom_attribute', explode('payments.payment.custom-attributes.custom-attribute'))
#flattenedPaymentDF = flattenedPaymentDF.withColumn('payment_custom_attribute_attribute_id', col('payment_custom_attribute._attribute-id'))
#flattenedDF = flattenedPaymentDF.withColumn('custom_attribute', explode('custom-attributes.custom-attribute'))
#.withColumn('payment_custom_attribute', explode('payments.payment.custom-attributes.custom-attribute')) \
#.withColumn('custom_attribute', explode('custom-attributes.custom-attribute'))

flattenedDF = inputDF.withColumn('product_lineitem', explode('product-lineitems.product-lineitem'))

orderDF = flattenedDF.withColumnRenamed('_order-no', 'orderNum') \
	.withColumnRenamed('order-date', 'orderDate') \
	.withColumnRenamed('created-by', 'createdBy') \
	.withColumnRenamed('original-order-no', 'originalOrderNum') \
	.withColumnRenamed('currency', 'currency') \
	.withColumnRenamed('customer-locale', 'customerLocation') \
	.withColumnRenamed('taxation', 'taxation') \
	.withColumnRenamed('invoice-no', 'invoiceNum') \
	.withColumnRenamed('remoteHost', 'remoteHost') \
	.withColumnRenamed('current-order-no', 'currentOrderNum') \
	.withColumn('customer_name', col('customer.customer-name')) \
	.withColumn('customer_email', col('customer.customer-email')) \
	.withColumn('billing_address_first_name', col('customer.billing-address.first-name')) \
	.withColumn('billing_address_last_name', col('customer.billing-address.last-name')) \
	.withColumn('billing_address_address1', col('customer.billing-address.address1')) \
	.withColumn('billing_address_address2', col('customer.billing-address.address2')) \
	.withColumn('billing_address_city', col('customer.billing-address.city')) \
	.withColumn('billing_address_postal_code', col('customer.billing-address.postal-code')) \
	.withColumn('billing_address_state_code', col('customer.billing-address.state-code')) \
	.withColumn('billing_address_country_code', col('customer.billing-address.country-code')) \
	.withColumn('billing_address_phone', col('customer.billing-address.phone')) \
	.withColumn('order_status', col('status.order-status')) \
	.withColumn('shipping_status', col('status.shipping-status')) \
	.withColumn('confirmation_status', col('status.confirmation-status')) \
	.withColumn('payment_status', col('status.payment-status')) \
	.withColumn('product_lineitem_net_price', col('product_lineitem.net-price')) \
	.withColumn('product_lineitem_tax', col('product_lineitem.tax')) \
	.withColumn('product_lineitem_gross_price', col('product_lineitem.gross-price')) \
	.withColumn('product_lineitem_base_price', col('product_lineitem.base-price')) \
	.withColumn('product_lineitem_lineitem_text', col('product_lineitem.lineitem-text')) \
	.withColumn('product_lineitem_tax_basis', col('product_lineitem.tax-basis')) \
	.withColumn('product_lineitem_position', col('product_lineitem.position')) \
	.withColumn('product_lineitem_product_id', col('product_lineitem.product-id')) \
	.withColumn('product_lineitem_product_name', col('product_lineitem.product-name')) \
	.withColumn('product_lineitem_quantity_unit', col('product_lineitem.quantity._unit')) \
	.withColumn('product_lineitem_quantity_VALUE', col('product_lineitem.quantity._VALUE')) \
	.withColumn('product_lineitem_tax_rate', col('product_lineitem.tax-rate')) \
	.withColumn('product_lineitem_shipment_id', col('product_lineitem.shipment-id')) \
	.withColumn('product_lineitem_gift', col('product_lineitem.gift')) \
	.withColumn('shipping_lineitem_net_price', col('shipping-lineitems.shipping-lineitem.net-price')) \
	.withColumn('shipping_lineitem_tax', col('shipping-lineitems.shipping-lineitem.tax')) \
	.withColumn('shipping_lineitem_gross_price', col('shipping-lineitems.shipping-lineitem.gross-price')) \
	.withColumn('shipping_lineitem_base_price', col('shipping-lineitems.shipping-lineitem.base-price')) \
	.withColumn('shipping_lineitem_lineitem_text', col('shipping-lineitems.shipping-lineitem.lineitem-text')) \
	.withColumn('shipping_lineitem_tax_basis', col('shipping-lineitems.shipping-lineitem.tax-basis')) \
	.withColumn('shipping_lineitem_item_id', col('shipping-lineitems.shipping-lineitem.item-id')) \
	.withColumn('shipping_lineitem_shipment_id', col('shipping-lineitems.shipping-lineitem.shipment-id')) \
	.withColumn('shipping_lineitem_tax_rate', col('shipping-lineitems.shipping-lineitem.tax-rate')) \
	.withColumn('shipment_shipment_id', col('shipments.shipment._shipment-id')) \
	.withColumn('shipment_shipping_status', col('shipments.shipment.status.shipping-status')) \
	.withColumn('shipment_shipping_method', col('shipments.shipment.shipping-method')) \
	.withColumn('shipment_shipping_address_first_name', col('shipments.shipment.shipping-address.first-name')) \
	.withColumn('shipment_shipping_address_last_name', col('shipments.shipment.shipping-address.last-name')) \
	.withColumn('shipment_shipping_address_address1', col('shipments.shipment.shipping-address.address1')) \
	.withColumn('shipment_shipping_address_address2', col('shipments.shipment.shipping-address.address2')) \
	.withColumn('shipment_shipping_address_city', col('shipments.shipment.shipping-address.city')) \
	.withColumn('shipment_shipping_address_postal_code', col('shipments.shipment.shipping-address.postal-code')) \
	.withColumn('shipment_shipping_address_state_code', col('shipments.shipment.shipping-address.state-code')) \
	.withColumn('shipment_shipping_address_country_code', col('shipments.shipment.shipping-address.country-code')) \
	.withColumn('shipment_shipping_address_phone', col('shipments.shipment.shipping-address.phone')) \
	.withColumn('shipment_gift', col('shipments.shipment.gift')) \
	.withColumn('shipment_merchandize_total_net_price', col('shipments.shipment.totals.merchandize-total.net-price')) \
	.withColumn('shipment_merchandize_total_tax', col('shipments.shipment.totals.merchandize-total.tax')) \
	.withColumn('shipment_merchandize_total_gross_price', col('shipments.shipment.totals.merchandize-total.gross-price')) \
	.withColumn('shipment_adjusted_merchandize_total_net_price', col('shipments.shipment.totals.adjusted-merchandize-total.net-price')) \
	.withColumn('shipment_adjusted_merchandize_total_tax', col('shipments.shipment.totals.adjusted-merchandize-total.tax')) \
	.withColumn('shipment_adjusted_merchandize_total_gross_price', col('shipments.shipment.totals.adjusted-merchandize-total.gross-price')) \
	.withColumn('shipment_shipping_total_net_price', col('shipments.shipment.totals.shipping-total.net-price')) \
	.withColumn('shipment_shipping_total_tax', col('shipments.shipment.totals.shipping-total.tax')) \
	.withColumn('shipment_shipping_total_gross_price', col('shipments.shipment.totals.shipping-total.gross-price')) \
	.withColumn('shipment_adjusted_shipping_total_net_price', col('shipments.shipment.totals.adjusted-shipping-total.net-price')) \
	.withColumn('shipment_adjusted_shipping_total_tax', col('shipments.shipment.totals.adjusted-shipping-total.tax')) \
	.withColumn('shipment_adjusted_shipping_total_gross_price', col('shipments.shipment.totals.adjusted-shipping-total.gross-price')) \
	.withColumn('shipment_shipment_total_net_price', col('shipments.shipment.totals.shipment-total.net-price')) \
	.withColumn('shipment_shipment_total_tax', col('shipments.shipment.totals.shipment-total.tax')) \
	.withColumn('shipment_shipment_total_gross_price', col('shipments.shipment.totals.shipment-total.gross-price')) \
	.withColumn('merchandize_total_net_price', col('totals.merchandize-total.net-price')) \
	.withColumn('merchandize_total_tax', col('totals.merchandize-total.tax')) \
	.withColumn('merchandize_total_gross_price', col('totals.merchandize-total.gross-price')) \
	.withColumn('adjusted_merchandize_total_net_price', col('totals.adjusted-merchandize-total.net-price')) \
	.withColumn('adjusted_merchandize_total_tax', col('totals.adjusted-merchandize-total.tax')) \
	.withColumn('adjusted_merchandize_total_gross_price', col('totals.adjusted-merchandize-total.gross-price')) \
	.withColumn('shipping_total_net_price', col('totals.shipping-total.net-price')) \
	.withColumn('shipping_total_tax', col('totals.shipping-total.tax')) \
	.withColumn('shipping_total_gross_price', col('totals.shipping-total.gross-price')) \
	.withColumn('adjusted_shipping_total_net_price', col('totals.adjusted-shipping-total.net-price')) \
	.withColumn('adjusted_shipping_total_tax', col('totals.adjusted-shipping-total.tax')) \
	.withColumn('adjusted_shipping_total_gross_price', col('totals.adjusted-shipping-total.gross-price')) \
	.withColumn('order_total_net_price', col('totals.order-total.net-price')) \
	.withColumn('order_total_tax', col('totals.order-total.tax')) \
	.withColumn('order_total_gross_price', col('totals.order-total.gross-price')) \
	.withColumn('payment_credit_card_card_type', col('payments.payment.credit-card.card-type')) \
	.withColumn('payment_credit_card_card_number', col('payments.payment.credit-card.card-number')) \
	.withColumn('payment_credit_card_card_holder', col('payments.payment.credit-card.card-holder')) \
	.withColumn('payment_credit_card_card_token', col('payments.payment.credit-card.card-token')) \
	.withColumn('payment_credit_card_expiration_month', col('payments.payment.credit-card.expiration-month')) \
	.withColumn('payment_credit_card_expiration_year', col('payments.payment.credit-card.expiration-year')) \
	.withColumn('payment_amount', col('payments.payment.amount')) \
	.withColumn('payment_processor_id', col('payments.payment.processor-id')) \
	.withColumn('payment_transaction_id', col('payments.payment.transaction-id')) \
	.withColumn('note_created_by', col('notes.note.created-by')) \
	.withColumn('note_creation_date', col('notes.note.creation-date')) \
	.withColumn('note_subject', col('notes.note.subject')) \
	.withColumn('note_text', col('notes.note.text')) \
	.drop('customer') \
	.drop('status') \
	.drop('product-lineitems') \
	.drop('shipping-lineitems') \
	.drop('shipments') \
	.drop('totals') \
	.drop('payments') \
	.drop('notes') \
	.drop('custom-attributes') \
	.drop('product_lineitem')
	
spark.sql("DROP TABLE IF EXISTS orders")
orderDF.write.saveAsTable("orders")
#orderDF.write.format("com.databricks.spark.csv").option("header", "true").mode("overwrite").save("adl://yetiadls.azuredatalakestore.net/clusters/yetidpe3600/orders_2006-10-31.csv")
#spark.sql("select * from orders limit 1)
