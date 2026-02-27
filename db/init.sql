-- init.sql
-- Generated from Adminer dump, reordered & made reproducible for local/dev.
-- MySQL 8.0+

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

SET time_zone = '+00:00';

-- =========================
-- Database (change name if you want)
-- =========================
CREATE DATABASE IF NOT EXISTS `mydb`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE `mydb`;

-- For reruns during local dev: drop in dependency-safe order
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS `category_corner_map`;
DROP TABLE IF EXISTS `product_discounts`;
DROP TABLE IF EXISTS `product_prices`;
DROP TABLE IF EXISTS `purchase_history`;
DROP TABLE IF EXISTS `store_corners`;
DROP TABLE IF EXISTS `products`;
DROP TABLE IF EXISTS `users`;

SET FOREIGN_KEY_CHECKS = 1;

-- =========================
-- Tables (dependency-safe order)
-- =========================

-- 1) products (referenced by product_prices)
CREATE TABLE `products` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `item_no` varchar(32) NOT NULL,
  `barcd` varchar(32) DEFAULT NULL,
  `product_name` varchar(255) NOT NULL,
  `company` varchar(128) DEFAULT NULL,
  `volume` varchar(64) DEFAULT NULL,
  `category_l` varchar(64) DEFAULT NULL,
  `category_m` varchar(64) DEFAULT NULL,
  `category_s` varchar(64) DEFAULT NULL,
  `nutrition_info` json DEFAULT NULL,
  `src_meta_xml` varchar(512) DEFAULT NULL,
  `dedup_key_type` varchar(32) DEFAULT NULL,
  `dedup_key` varchar(64) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `picture` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_products_barcd` (`barcd`),
  KEY `idx_products_item_no` (`item_no`),
  KEY `idx_products_product_name` (`product_name`),
  KEY `idx_products_category` (`category_l`,`category_m`,`category_s`),
  KEY `idx_products_company` (`company`),
  FULLTEXT KEY `ft_products_name_company` (`product_name`,`company`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 2) users (referenced by purchase_history)
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password_hash` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `name` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `role` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `is_active` tinyint(1) DEFAULT NULL,
  `created_at` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ix_users_username` (`username`),
  KEY `ix_users_id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3) store_corners (referenced by category_corner_map)
CREATE TABLE `store_corners` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `corner_no` int NOT NULL,
  `corner_name` varchar(100) DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_store_corners_corner_no` (`corner_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 4) product_prices (references products)
CREATE TABLE `product_prices` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `product_id` bigint unsigned NOT NULL,
  `price` int NOT NULL,
  `currency` char(3) NOT NULL DEFAULT 'KRW',
  `source` varchar(64) DEFAULT NULL,
  `checked_at` datetime(6) NOT NULL,
  `query_type` varchar(32) DEFAULT NULL,
  `query_value` varchar(255) DEFAULT NULL,
  `mall_name` varchar(128) DEFAULT NULL,
  `match_title` varchar(512) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_product_prices_snapshot` (`product_id`,`checked_at`,`source`,`price`),
  KEY `idx_product_prices_product_checked` (`product_id`,`checked_at` DESC),
  KEY `idx_product_prices_checked_at` (`checked_at` DESC),
  KEY `idx_product_prices_source_checked` (`source`,`checked_at` DESC),
  CONSTRAINT `fk_product_prices_product_id`
    FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 5) product_discounts (references product_prices)
CREATE TABLE `product_discounts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `product_price_id` bigint unsigned NOT NULL,
  `is_discounted` tinyint(1) NOT NULL DEFAULT '0',
  `discount_rate` decimal(5,2) DEFAULT NULL,
  `discount_amount` int DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_product_discounts_price` (`product_price_id`),
  KEY `idx_product_discounts_is_discounted` (`is_discounted`),
  CONSTRAINT `fk_product_discounts_price`
    FOREIGN KEY (`product_price_id`) REFERENCES `product_prices` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6) category_corner_map (references store_corners)
CREATE TABLE `category_corner_map` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `category_l` varchar(255) NOT NULL,
  `corner_id` bigint NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_category_corner_map_category_l` (`category_l`),
  KEY `idx_category_corner_map_corner_id` (`corner_id`),
  CONSTRAINT `fk_category_corner_map_corner`
    FOREIGN KEY (`corner_id`) REFERENCES `store_corners` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 7) purchase_history (references users)
CREATE TABLE `purchase_history` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `items` json NOT NULL,
  `total_amount` int NOT NULL,
  `timestamp` datetime DEFAULT NULL,
  `notes` text COLLATE utf8mb4_unicode_ci,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `ix_purchase_history_id` (`id`),
  CONSTRAINT `purchase_history_ibfk_1`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;